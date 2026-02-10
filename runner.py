import time
import threading
import logging
import importlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_module(module_name: str):
    while True:
        try:
            logging.info("Starting module: %s", module_name)
            mod = importlib.import_module(module_name)

            if hasattr(mod, "main") and callable(mod.main):
                mod.main()
            else:
                logging.info("%s has no main(); keeping thread alive.", module_name)
                while True:
                    time.sleep(3600)

        except Exception as e:
            logging.exception("Module %s crashed: %s. Restarting in 30s...", module_name, e)
            time.sleep(30)


def main():
    threads = [
        threading.Thread(target=run_module, args=("ichimokubot",), daemon=True),
        threading.Thread(target=run_module, args=("pagibig_scanner",), daemon=True),
    ]

    for t in threads:
        t.start()

    logging.info("Runner started both ichimokubot + pagibig_scanner.")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
