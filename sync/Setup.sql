CREATE OR REPLACE FUNCTION setup_table_from_sync(
    table_name TEXT,
    create_sql TEXT
)
RETURNS TEXT AS $$
BEGIN
    -- Drop the old table if it exists
    EXECUTE 'DROP TABLE IF EXISTS "' || table_name || '";';
    
    -- Execute the new CREATE TABLE statement
    EXECUTE create_sql;
    
    RETURN 'Table ' || table_name || ' successfully reset.';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;