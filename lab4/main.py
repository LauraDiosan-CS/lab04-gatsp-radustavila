from start.UI import UI
from start.repo import Repo
from start.service import Service


repo = Repo()
service = Service(repo)
ui = UI(service)
ui.start()
