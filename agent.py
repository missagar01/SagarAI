import os
import json
from typing import List, TypedDict
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


# --- Pydantic Models for Intent Classification ---
class DatabaseQuery(BaseModel):
    """The user is asking a question that requires a database query, or can be solved by an sql query"""

    pass


class Conversation(BaseModel):
    """The user is greeting, making a small talk or asking a general knowledge question not related to database or cannot be handled with sql."""

    pass

# --- Pydantic Models for Database Selection ---
class CompanyDatabase(BaseModel):
    """The user query is related to tasks, checklist, delegation, employees of the company (users of the database)."""

    pass

class SalesDatabase(BaseModel):
    """The user query is related to purchase orders, store indents, vendors, items in store, deliveries, receipts invoices.
       The user mentions tables names like store, o2d, enquiry, lead etc.
    """

    pass


# --- Task Definition ---
class Task(TypedDict):
    description: str
    timestamp: str
    planned_date: str
    department: str


# --- Agent State Definition ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    query: str
    result: str
    retries: int
    database: str

    task_details: Task

    answer: str
    intent: str


sales_db_uri = os.getenv("SALES_DATABASE_URI")
sales_engine = create_engine(sales_db_uri)
sales_db = SQLDatabase(engine=sales_engine)
sales_execute_query_tool = QuerySQLDatabaseTool(db=sales_db)

company_db_uri = os.getenv("COMPANY_DATABASE_URI")
company_engine = create_engine(company_db_uri)
company_db = SQLDatabase(engine=company_engine)
company_execute_query_tool = QuerySQLDatabaseTool(db=company_db)
llm = ChatOpenAI(model="gpt-4.1", temperature=0)


@tool
def get_current_datetime() -> str:
    """Returns today's date and the current time in ISO 8601 format."""
    return datetime.now().isoformat()


# @tool
# def get_datetime_from_query(query: str) -> str:
#     """Returns a date time in ISO 8601 format based on the query in english and suitable for parsing date time with dateparser."""

#     return dateparser.parse(
#         query, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now()}
#     ).isoformat()


# --- Graph Nodes ---
def classify_intent_node(state: AgentState):
    """Classifies the user's question by forcing the LLM to call a specific tool."""
    print("--- Classifying Intent (with Function Calling) ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intent classifier. Call the appropriate tool based on the user's last message.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tools = [DatabaseQuery, Conversation]
    llm_with_tools = llm.bind_tools(tools)
    runnable = prompt | llm_with_tools
    ai_message = runnable.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        }
    )
    intent = (
        "Conversation"
        if not ai_message.tool_calls
        else ai_message.tool_calls[0]["name"]
    )
    print(f"Intent: {intent}")
    return {"intent": intent}


def handle_conversation_node(state: AgentState):
    """Creates natural conversation with the user."""
    print("--- Handling Conversation ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant solves user's database related queries, Diya, for Mr. Akash Agrawal, you refer him as Akash Sir. Reply to the user politely with a short relevant relevant response. Reply in English or Hindi based on user's question. All currencies are in Rupees until mentioned other wise. Greet user according to current time, i.e., 'Good Morning', 'Good Evening', etc. when needed. Don't just greet on every response. Show the units when needed.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    tools = [get_current_datetime]
    agent_runnable = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)
    result = agent_executor.invoke(
        {"question": state["question"], "chat_history": state.get("chat_history", [])}
    )
    print(f"Final Answer: {result['output']}")
    return {"answer": result["output"]}

def select_database_node(state: AgentState):
    """Selects the appropriate database based on user's question."""
    print("--- Selecting Database (with Function Calling) ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a database classifier. Call the appropriate tool based on the user's last message.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    tools = [CompanyDatabase, SalesDatabase]
    llm_with_tools = llm.bind_tools(tools)
    runnable = prompt | llm_with_tools
    ai_message = runnable.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
        }
    )
    database = (
        "CompanyDatabase"
        if not ai_message.tool_calls
        else ai_message.tool_calls[0]["name"]
    )
    print(f"Database: {database}")
    return {"database": database}


def generate_query_node(state: AgentState):
    """Takes the user's question, generates a SQL query, and adds it to the state."""
    print("--- Generating SQL Query ---")
    sales_description = """
    - When a user asks about "orders" or "po", they are usually referring to entries where a table has fields relevant to Purchase Orders like "Quantity", "PO Number" or "Indent Number".
    - When a user refers to sheets they are actually talking about tables.
    - Here is a list of tables that are present in the database along with their descriptions:
        - **O2D:** 
            - This table contains data related to orders to delivery.
            - The steps it records are
                - Gate Entry of Vehicles: Timestamp, Order Number, Gate Entry Number, Customer Name, Truch Number 
                - First Wighment: Planned 2 (when the first weighment is planned), Actual 2 (when the first weighment actually happened), Time Delay 2 (delay in first weighment), WB Slip No (weighbridge slip number for first weighment)
                - Load vehicle as per loading slip: Planned 3 (when loading is planned), Actual 3 (when loading actually happened), Time Delay 3 (delay in loading), Supervisor Name (name of the supervisor overseeing loading), Remarks (any remarks about loading)
                - Second Weighment: Planned 4 (when the second weighment is planned), Actual 4 (when the second weighment actually happened), Time Delay 4 (delay in second weighment), Final Weight (final weight recorded during second weighment) 
                - Generate Invoice: Planned 5 (when invoice generation is planned), Actual 5 (when invoice is actually generated), Time Delay 5 (delay in invoice generation), Invoice Number (invoice number generated), Invoice Date (date of invoice generation), Broker Name (name of the broker involved), Sales Person (name of the salesperson handling the order), Customer Name (name of the customer for the order), Loaded Truck Number (truck number used for loading), Item Name (name of the item being delivered), Quantity (quantity of product delivered), Amount (total amount for the order), State (state where delivery is made)
                - Gate Out: Planned 6 (when gate out is planned), Actual 6 (when gate out actually happened), Time Delay 6 (delay in gate out)
                - Payment Reminder: Planned 7 (when payment reminder is planned), Actual 7 (when payment reminder actually happened), Time Delay 7 (delay in payment reminder), Total Received (total payment received), Balance Amount (remaining balance amount)
        - **STORE:**
            - This table contains data for Purchase And Issue orders of the store.
            - The steps it records are
                - Generate Store Indent: Timestamp, Indent Number, Indenter Name, Department, Group Head (Category of Item), Item Code, Quantity, UOM, Specifications, Indent Approved By, Indent Type (Purchase or Store Out), Attachments (optional)
                - Vendor Type Selection: Planned 1, Actual 1, Time Delay 1, Vendor Type (Three Party or Regular), Approved Quantity
                - Three party vendor filling: Planned 2, Actual 2, Time Delay 2 and, Vendor Name X, Rate X and Payment Term X for X 1 to 15 vendors and An optional Comparison Sheet attchement at the end (no need to bother with this).
                - Vendor Approval, approved vendor details from regular vendors or Selected vendor from three party: Planned 3, Actual 3, Time Delay 3, Approved Vendor Name, Approved Rate, Approved Payment Term, Approved Date. 
                - Purchase Order: Planned 4, Actual 4, Time Delay 4, PO Number, PO copy (attachment, no need to bother).
                - Received In Store: Planned 5, Actual 5, Time Delay 5, [Receive Status is useless].
                - Issue From Store: Planned 6, Actual 6, Time Delay 6, Issue Is Approved By, Issued Quantity, Issue Status.
            - Regular vendors are old company vendors, where three party vendors are new vendors suggested.
        - **LEAD:**
            - This table tracks the complete customer acquisition and conversion journey in three steps — from initial contact to final order.This table contains data related to sales leads and their follow ups.
            - Step 1: Lead Creation (Customer Contacts Us) - This step records how a new customer reaches out.
                - Timestamp - Date and time the lead was created.
                - LD-Lead No. - Unique number identifying each lead.
                - Lead Receiver Name - Executive or staff member who first received the enquiry.
                - Lead Source - How the customer contacted (e.g., website, referral, phone, ad).
                - Company Name / Customer Details - Customer or company name and contact info.
                - Phone Number / Email Address - Communication details.
                - Location / State / Address - Where the customer is located.
                - Salesperson Name - Salesperson assigned to handle the lead.
                - NOB (Nature of Business) - Type or sector of the customer’s business.
                - Additional Notes - Any background details shared during first contact.
            - Step 2: Lead Call (We Contact the Customer) - In this step, a sales executive calls the lead to understand their requirements.
                - Planned / Actual / Delay - Planned vs. actual time of the lead call.
                - Status - Whether the call was completed, missed, or pending.
                - What did the customer say? - Notes from the conversation (requirements, interest, etc.).
                - Enquiry Received Status / Date / Approach - Whether an enquiry was confirmed, when, and how.
                - Project Approximate Value - Estimated value of the customer's potential order.
                - Item / Qty / Total Qty - Products or quantities discussed.
                - Next Action - Follow-up task after the call (e.g., send quotation, schedule next call).
                - Next Call Date / Time - When the next contact with the customer is planned.
            - Stet 3: Enquiry Follow-Up & Order Creation (Final Conversion) - Here the executive follows up again, sends quotations, and finalizes the order.
                - Planned1 / Actual1 / Delay - Scheduled vs. actual follow-up call timings.
                - Enquiry Status - Stage of the enquiry (e.g., in progress, converted, lost).
                - What Did Customer Say / Current Stage - Customer response and deal progress.
                - Quotation Details - Includes quotation number, sender, and amounts (with/without tax).
                - Quotation Upload / Remarks - Attached quotation file and notes.
                - Follow-up Status / Next Call Date / Time - Ongoing follow-up tracking.
                - Is Order Received? / Acceptance Via - Whether an order was received and how (email, PO, etc.).
                - Payment Mode / Terms / Transport Mode - Payment and delivery details.
                - PO Number / Acceptance File Upload - Purchase order references and documents.
                - Remarks / Hold Reasons / Dates - For orders on hold or not yet confirmed.
                - SC Name - Sales coordinator managing the process.
        - **ENQUIRY:**
            - This table tracks existing customer enquiries that come directly for new orders.
            - Step 1 — Enquiry Received (Customer Contacts Us Again) - This step logs all details when a returning or existing customer reaches out with a new enquiry.
                - Timestamp - Date and time when the enquiry record was created.
                - En-Enquiry No. - Unique identification number for the enquiry.
                - Lead Source - How the enquiry came in (call, email, visit, reference, etc.).
                - Company Name - Customer`s company name.
                - Phone Number / Email - Customer contact information.
                - Sales Person Name - Sales executive responsible for this enquiry.
                - Location - City or area of the customer.
                - Enquiry Receiver Name - Person who logged or received the enquiry.
                - Enquiry Date - Date when the enquiry was received.
                - Enquiry Approach - Medium through which the customer reached out.
                - Item / Qty - Products or services requested and their quantities.
            - Step 2 — Follow-Up, Quotation & Order Creation - Covers the process of contacting the customer, sending quotations, and completing the order.
                - Planned / Actual / Delay - Scheduled and actual follow-up timings, plus any delay.
                - Enquiry Status - Current stage of the enquiry (new, in progress, converted, lost).
                - What Did Customer Say - Feedback or response during the follow-up.
                - Current Stage - Position in the sales cycle (enquiry, quotation, negotiation).
                - Send Quotation No. / Quotation Number - Quotation identifiers.
                - Quotation Shared By - Salesperson who sent the quotation.
                - Quotation Value Without Tax / With Tax - Quotation amount details.
                - Quotation Upload / Quotation Remarks - Quotation document and any remarks.
                - Follow-Up Status / Next Call Date / Next Call Time - Schedule for continued communication.
                - Is Order Received? Status / Acceptance Via - Whether the customer confirmed the order and how (email, PO, verbal).
                - Payment Mode / Payment Terms (In Days) - Payment details and terms.
                - Transport Mode - Mode of delivery or dispatch.
                - PO Number / Acceptance File Upload - Purchase order reference and file upload.
                - Remark - General notes or summary.
                - If No Then Get Relevant Reason Status / Remark - Reason if the order was not received.
                - Customer Order Hold Reason Category / Holding Date / Hold Remark - If the order was put on hold and why.
                - Sales Coordinator Name - Person monitoring the enquiry process.
    - When user asks query based on some identity, that can be present in other tables, and there is no previous context for choosing a table, give data, or all occurances.
    - Usually a pending item is when its planned is filled but its actual is empty or null. 
    """

    company_description = """
    - When a user asks about "tasks" or "kaam", they are referring to entries where a table has fields relevant to tasks, like "Checklist" and "Delegation". You MUST query one of given tables that is related to tasks. DO NOT invent or query a non-existent table named "tasks".
    - Here is a list of tables that are present in the database that you need to care about along with their descriptions:
        - checklist:
            - This table contains data related to tasks assigned to employees.
            - The task is completed if Status is "YeS" and submission_date is filled.
            - If status is "No" or Empty and submission_data is filled, the task has ended but not completed.
            - If status is "No" or Empty and submission_date is empty, the task is pending.
            - Name is the assignee of the task.
            - given_by is the assigner of the task.
            - Rest of columns are pretty self explanatory.
        - delegation:
            - These tasks with frequency of one-time are stored here.
            - The 'color_code' column is used to store the number of times it to took to complete the task.
            - Rest is pretty self explanatory.
        - users:
            - This table contains data related to users of the checklist and delegation.
            - The leave_date marks the start of leave for an user.
            - The leave_end_date marks the end of leave for an user.
            - And remark is the reason for leave. 
    - Don't user any other table except checklist, delegation and users for answering task related queries.
    - When user asks query based on some identity, that can be present in other tables, and there is no previous context for choosing a table, give data, or all occurances.
    """

    system_prompt = """You are an AI expert in writing PostgreSQL queries.
    Given a user question and conversation history, create a syntactically correct PostgreSQL query.
    The query should fullfill user's query.
    The query should work on the given schema.
    {schema}

    --- Querying Rules ---
    1.  **CRITICAL `UNION` RULE:** When using `UNION` or `UNION ALL`, you **MUST NOT** use `SELECT *`. The tables have different columns and this will cause an error.
    2.  **HOW TO FIX `UNION`:** You must explicitly list the columns to select. Identify a set of common, meaningful columns (e.g., "Task", "Status", "Assignee", "Priority", "Due_Date"). For tables that are missing one of these columns, you **MUST** select `NULL` and cast it to the appropriate type, aliasing it to the common column name. For example: `SELECT "Task", "Status", NULL::text AS "Assignee" FROM "Checklist"`.
    3. Use advanced matching techniques, to respond to more flexible queries.

    --- Database Descriptions ---
    {database_description}
    -----------------------------
    
    --- Data Dictionary ---
    - The "Status" column: 'Completed', 'Yes', 'Done' all mean the task is complete. NULL/Empty, 'Not Complete', 'Pending' may mean the task is pending. Basically anything not complete is pending.
    - The "Priority" column: 'High', 'Urgent', 'H' all mean high priority. 'Low' and 'L' mean low priority.
    -----------------------

    --- Instructions ---
    - Report queries should include
        - Total number of relevant entries
        - Total amount pending (if applicable)
        - Total completed (if applicable)
        - Total pending (if applicable)
        - Other relevant data points based on the columns in the table.
        - Not all data points are directly available from columns names, some data points need to be generated using SQL functions like COUNT, SUM, etc. on relevant columns.
        - And a small table with aggregate data based on given by, vendors or parties and there products etc. showing insight on the data. Though data points are more important.
        - Calculate quantities, amounts, etc. based on different columns in the table. For example, 
            - total amount pending can be calculated using SUM of "Amount" column where "Status" is 'Pending'.
            - total quantity can be calculated using different columns of the row related to quantity like "Quantity", "Total Lifted", "Order Cancelled Quantity", etc. ex Pending Quantity = Quantity - Total Lifted Quantity.
            - Make sure to calculate these data not just SUM("COLUMN_NAME") everywhere.
            - Show all the relevant columns in the final table.
    - Make sure that the output of SQL query gives all data at once. Only give one query.
    - Make sure to query in limits, as there is a lot of data in the tables. And the limits should make sense for the question asked.
    - The limits should give wrong data for the user's query. Still it should not query 1000s of rows.
    --------------------
    - **IMPORTANT:** Add comments in the SQL query to explain your logic wherever possible.
    - **IMPORTANT:** Only return the SQL query. Do not add any other text or explanation.
    - **IMPORTANT:** If a table or column name contains a space or is a reserved keyword, you MUST wrap it in double quotes. For example: "Task Description".
    - **IMPORTANT:** Use the columns provided in the schema, if user mention a column that is not in schema, try to find the closest relevant column in the schema.
    """


    if "Error:" in state.get("result", ""):
        system_prompt += """
        \n---
        The previous query you wrote failed. Here is the error message: 
        {error}
        You likely violated the CRITICAL UNION RULE. Do not use SELECT *. Instead, select specific columns and use NULL placeholders for columns that do not exist in some tables. Please write a new corrected SQL query.
        ---
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | llm
    raw_query = runnable.invoke(
        {
            "question": state["question"],
            "chat_history": state.get("chat_history", []),
            "schema": state["database"] == "SalesDatabase" and sales_db.get_table_info() or company_db.get_table_info(),
            "error": state.get("result", ""),
            "database_description": state["database"] == "SalesDatabase" and sales_description or company_description,
        }
    ).content
    sql_query = raw_query.strip().replace("```sql", "").replace("```", "").strip()
    print(f"Generated Query: {sql_query}")
    retries = state.get("retries", 0)
    return {"query": sql_query, "retries": retries + 1}


def execute_query_node(state: AgentState):
    """Executes the SQL query and returns the result."""
    print("--- Executing SQL Query ---")
    query = state["query"]
    if state["database"] == "SalesDatabase":
        execute_query_tool = sales_execute_query_tool
    else:
        execute_query_tool = company_execute_query_tool
    result = execute_query_tool.invoke(query)
    print(f"Query Result: {result}")
    return {"result": result}


def summarize_result_node(state: AgentState):
    """Takes the query result and creates a natural language answer."""

    system_prompt = """
    You are a helpful AI assistant, Diya. 
    Your job is to answer the user's question in concise manner, based on the data provided, which should be easy and fast to read, with markup and lists and tables if needed. 
    Only reply in English or Hindi based on user's question. 
    Do not give any clarification about how you got the result. 
    Never reply with more than 20 rows of data, whether that be in list or tables.
    Show data points in readable format.
    All currencies are in Rupees until mentioned otherwise. Show the relevant units wherever possible.
    Keep the large numbers in human readable format, and use indian number system (lakhs, crores) and commas.
    In reports, based on data points, give bite sized insights on the data. Bold the important numbers and details.
    Show information related to all rows seprately, if needed use tables or lists in reports.
    The structure of report should be,
        1. The table or list of data (if applicable)
        2. The data points summary
        3. The insights on the data.
    **Not all queries are reports, some are simple questions, answer them concisely. So don't follow the above format for normal answers**
    Don't user words like "database", "tables", "SQL query" etc. in your final answer.
    """

    print("--- Summarizing Result ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """Based on the user's question: "{question}"
        The following SQL query was generated: "{query}"
        And here is the result from the database: "{result}"
        Please provide a clear, natural language answer.
        Normalize table names, and remove _ in between words.
        """,
            ),
        ]
    )
    runnable = prompt | llm
    answer = runnable.invoke(
        {
            "question": state["question"],
            "query": state["query"],
            "result": state["result"],
        }
    ).content
    print(f"Final Answer: {answer}")
    return {"answer": answer}


def handle_error_node(state: AgentState):
    """Handles cases where the agent gives up after multiple retries."""
    print("--- 😩 Agent failed after multiple retries ---")
    error_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, Diya, for a SQL database. The query you generated failed multiple times. Just say to the user that you couldn't find the answer. Resturn small easy to read with markup response. All currencies are in Rupees until mentioned otherwise. Show the units wherever possible.",
            ),
            (
                "human",
                """The user asked: "{question}"
        Your last attempted SQL query was: "{query}"
        It failed with the error: "{error}"
        Please provide a clear, natural language response apologizing for the failure and offering advice.""",
            ),
        ]
    )
    runnable = error_prompt | llm
    answer = runnable.invoke(
        {
            "question": state["question"],
            "query": state["query"],
            "error": state.get("result", "Unknown error"),
        }
    ).content
    print(f"Final Answer: {answer}")
    return {"answer": answer}


# --- Conditional Edges ---
def decide_intent_path(state: AgentState):
    if state["intent"] == "DatabaseQuery":
        return "generate_query"
    return "handle_conversation"


def decide_result_status(state: AgentState):
    if "Error:" in state["result"]:
        print("--- Query failed. Looping back to generate a new query. ---")
        return "handle_error" if state["retries"] > 7 else "generate_query"
    return "summarize_result"


def decide_response_type(state: AgentState):
    """Analyzes the current state, and based not details, creates new task, or asks followup questions."""

    description = state["task_details"].get("description", None)
    timestamp = state["task_details"].get("timestamp", None)
    planned_date = state["task_details"].get("planned_date", None)
    department = state["task_details"].get("department", None)

    if not all((description, timestamp, planned_date, department)):
        return "ask_followup"
    return "create_task"


# --- Build the Graph ---
graph = StateGraph(AgentState)
graph.add_node("classify_intent", classify_intent_node)
graph.add_node("handle_conversation", handle_conversation_node)
graph.add_node("select_database", select_database_node)
graph.add_node("generate_query", generate_query_node)
graph.add_node("execute_query", execute_query_node)
graph.add_node("summarize_result", summarize_result_node)
graph.add_node("handle_error", handle_error_node)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges(
    "classify_intent",
    decide_intent_path,
    {
        "generate_query": "select_database",
        "handle_conversation": "handle_conversation",
    },
)
graph.add_edge("select_database", "generate_query")
graph.add_edge("generate_query", "execute_query")
graph.add_conditional_edges(
    "execute_query",
    decide_result_status,
    {
        "generate_query": "generate_query",
        "summarize_result": "summarize_result",
        "handle_error": "handle_error",
    },
)
graph.add_edge("handle_conversation", END)
graph.add_edge("handle_error", END)
graph.add_edge("summarize_result", END)

agent = graph.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    # Example question
    initial_state = {
        "question": "How many tasks are pending in the delegation list?",
        "chat_history": [],
    }
    final_state = agent.invoke(initial_state)
    print("\n--- Final State ---")
    print(final_state["answer"])
