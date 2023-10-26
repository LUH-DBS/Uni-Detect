install:
	conda env create --file environment.yml

uninstall:
	conda remove --name Uni-Detect --all

fd-offline:
	conda run --no-capture-output -n Uni-Detect python fd_violations/run_offline_learning_fd.py fd_violations/config/offline_fd_config.yml

fd-test:
	conda run --no-capture-output -n Uni-Detect python fd_violations/test_fd.py fd_violations/config/test_fd_config.yml

fd-test-mp:
	conda run --no-capture-output -n Uni-Detect python fd_violations/test_fd_mp.py fd_violations/config/test_fd_config.yml

no-offline:
	conda run --no-capture-output -n Uni-Detect python numeric_outliers/run_offline_learning_no.py numeric_outliers/config/offline_no_config.yml

no-test:
	conda run --no-capture-output -n Uni-Detect python numeric_outliers/test_no.py numeric_outliers/config/test_no_config.yml

se-offline:
	conda run --no-capture-output -n Uni-Detect python nspelling/run_offline_learning_se.py spelling/config/offline_se_config.yml

se-test:
	conda run --no-capture-output -n Uni-Detect python spelling/test_se.py spelling/config/test_se_config.yml

se-test-mp:
	conda run --no-capture-output -n Uni-Detect python spelling/test_se_mp.py spelling/config/test_se_config.yml

se-test-autodetectdata:
	conda run --no-capture-output -n Uni-Detect python spelling/test_se_autodetectdata.py spelling/config/test_se_config.yml

uv-offline:
	conda run --no-capture-output -n Uni-Detect python uniqueness_violations/run_offline_learning_uv.py uniqueness_violations/config/offline_uv_config.yml

uv-test:
	conda run --no-capture-output -n Uni-Detect python uniqueness_violations/test_uv.py uniqueness_violations/config/test_uv_config.yml

.PHONY: install, uninstall, fd-offline, fd-test, fd-test-mp, no-offline, no-test, se-offline, se-test, se-test-mp, se-test-autodetectdata, uv-offline, uv-test