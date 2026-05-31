from bs4 import BeautifulSoup
import time
import random
import logging
from pathlib import Path
import os
from curl_cffi import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

TABLES_DIR = Path("./dataset/tables")
if not TABLES_DIR.exists():
    os.makedirs(TABLES_DIR)


div_id_old = "all_leading_all_nba"
div_id_rookie = "all_leading_all_rookie"

years = list(range(1990, 2027))
random.shuffle(years)

for year in years:
    html_path_old = TABLES_DIR.joinpath(f"all_nba_{year}.html")
    html_path_rookie = TABLES_DIR.joinpath(f"all_nba_{year}_rookie.html")

    if html_path_old.exists() and html_path_rookie.exists():
        logger.info(f"HTML file for year {year} already exists. Skipping fetch.")
        continue

    url = f"https://www.basketball-reference.com/awards/awards_{year}.html"

    logger.info(f"Fetching data for year {year} from {url}.")

    response = requests.get(url, impersonate="chrome120")

    if response.status_code == 200:
        logger.info(f"Successfully fetched data for year {year}.")
    elif response.status_code == 429:
        logger.warning(
            f"Received 429 Too Many Requests for year {year}. Waiting before retrying..."
        )
        time.sleep(random.uniform(61, 71))
    else:
        logger.error(
            f"Failed to fetch data for year {year}. Status code: {response.status_code}"
        )
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    div_with_table_old = soup.find("div", id=div_id_old)
    if div_with_table_old:
        logger.info(f"Found div with id '{div_id_old}' for year {year}.")
        table = div_with_table_old.find("table")
        if table:
            logger.info(f"Found table in div for year ALL-NBA {year}.")
            with open(html_path_old, "w", encoding="utf-8") as f:
                f.write(str(table))
            logger.info(f"Saved table ALL-NBA HTML for year {year} to {html_path_old}.")
        else:
            logger.info(f"No table ALL-NBA found in div for year {year}.")
    else:
        logger.info(f"No div with id '{div_id_old}' found for year {year}.")



    div_with_table_rookie = soup.find("div", id=div_id_rookie)
    if div_with_table_rookie:
        logger.info(f"Found div with id '{div_id_rookie}' for year {year}.")
        table = div_with_table_rookie.find("table")
        if table:
            logger.info(f"Found table in div for year ALL-ROOKIE {year}.")
            with open(html_path_rookie, "w", encoding="utf-8") as f:
                f.write(str(table))
            logger.info(
                f"Saved table ALL-ROOKIE HTML for year {year} to {html_path_rookie}."
            )
        else:
            logger.info(f"No table ALL-ROOKIE found in div for year {year}.")

    else:
        logger.info(f"No div with id '{div_id_old}' found for year {year}.")
    logger.info("=" * 50)
    time.sleep(random.uniform(6, 7))
