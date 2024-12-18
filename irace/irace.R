library(irace)

Sys.setenv(PATH = paste("/home/pedro/Repositories/tcc-maintenance-planning-problem/.venv/bin/python", Sys.getenv("PATH"), sep = ":"))

scenario <- readScenario("scenario.txt")

irace(scenario = scenario)