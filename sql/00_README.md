# Zetaris setup scripts for whybot

Run them in order with the getting-started runner, `--dry-run` first:

    cd ../../zetaris-getting-started   # repo with scripts/run_sql.py, .env.local
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
    for f in 01_noaa_nyc_create.sql 02_us_holidays_create.sql 03_whybot_views_create.sql; do
      .venv/bin/python scripts/run_sql.py ../whybot/sql/$f --channel jdbc --skip-exists
    done
    # after a driver restart only:
    .venv/bin/python scripts/run_sql.py ../whybot/sql/04_cache_refresh.sql --channel jdbc

`10_federated_daily_select.sql` is the read-only federated query (taxi x weather x holidays x TPCDS date_dim).
See ../docs/zetaris.md for data coverage and caveats.
