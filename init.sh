#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until psql -U postgres -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

# Create application user and database
psql -U postgres <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
            CREATE ROLE ${POSTGRES_USER} WITH LOGIN PASSWORD '${POSTGRES_PASSWORD}';
        END IF;
    END \$\$;
    
    CREATE DATABASE ${POSTGRES_DB} WITH OWNER ${POSTGRES_USER};
    GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};
    
    -- Create and configure schema
    \c ${POSTGRES_DB}
    CREATE SCHEMA IF NOT EXISTS drone_schema AUTHORIZATION ${POSTGRES_USER};
    GRANT ALL PRIVILEGES ON SCHEMA drone_schema TO ${POSTGRES_USER};
    ALTER ROLE ${POSTGRES_USER} SET search_path TO drone_schema, public;
    
    -- Secure the public schema
    REVOKE ALL ON SCHEMA public FROM PUBLIC;
    GRANT USAGE ON SCHEMA public TO ${POSTGRES_USER};
EOSQL

# Wait a bit to ensure everything is ready
sleep 2
#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until psql -U postgres -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

# Create application user and database
psql -U postgres <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
            CREATE ROLE ${POSTGRES_USER} WITH LOGIN PASSWORD '${POSTGRES_PASSWORD}';
        END IF;
    END \$\$;
    
    CREATE DATABASE ${POSTGRES_DB} WITH OWNER ${POSTGRES_USER};
    GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};
    
    -- Create and configure schema
    \c ${POSTGRES_DB}
    CREATE SCHEMA IF NOT EXISTS drone_schema AUTHORIZATION ${POSTGRES_USER};
    GRANT ALL PRIVILEGES ON SCHEMA drone_schema TO ${POSTGRES_USER};
    ALTER ROLE ${POSTGRES_USER} SET search_path TO drone_schema, public;
    
    -- Secure the public schema
    REVOKE ALL ON SCHEMA public FROM PUBLIC;
    GRANT USAGE ON SCHEMA public TO ${POSTGRES_USER};
EOSQL

# Wait a bit to ensure everything is ready
sleep 2
psql -U postgres -d ${POSTGRES_DB} <<-EOSQL
    -- Change ownership
    DO \$\$
    BEGIN
        EXECUTE (
            SELECT string_agg(format('ALTER TABLE public.%I OWNER TO postgres', tablename), '; ')
            FROM pg_tables
            WHERE schemaname = 'public'
        );
    END \$\$;

    -- Grant permissions to application user
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${POSTGRES_USER};
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${POSTGRES_USER};
EOSQL