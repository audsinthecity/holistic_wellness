from dotenv import load_dotenv
load_dotenv()

import os
HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from crewai import Crew, Process

from tasks import ReadWriteTasks
from agents import ReadWriteAgents

tasks = ReadWriteTasks()
agents = ReadWriteAgents()

print("## Welcome to the Research and Write Crew")
print('-------------------------------')

# Create Agents
researcher_agent = agents.research_agent()
writer_agent = agents.writer_agent()

# Create Tasks
research = tasks.research_task(researcher_agent)
write = tasks.write_task(writer_agent)

# Create Crew responsible for Copy
crew = Crew(
	agents=[
		researcher,
		writer,
	],
	tasks=[
		research_task,
		write_task
	],
  process=Process.sequential, #sequential is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

result = crew.kickoff(inputs={'topic': Best functional health suggestions to cause hormesis'})


# Print results
print("\n\n################################################")
print("## Here is the result")
print("################################################\n")
print(result)
