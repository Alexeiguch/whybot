# make -C playground <target>   (run from the repo root)
.PHONY: playground local docker test
playground: local
local:            ## run on this machine (creates .venv on first run)
	./run.sh
docker:           ## run in Docker
	docker compose -f docker-compose.yml up --build
test:             ## smoke tests (SAMPLE_MODE, offline)
	cd .. && $${PYTHON:-$$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )} -m pytest playground/tests -q
