# scrap all players
from nba_api.stats.endpoints import commonallplayers


players = commonallplayers.CommonAllPlayers().get_data_frames()
