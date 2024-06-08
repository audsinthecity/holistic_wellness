from dotenv import load_dotenv
load_dotenv()

# Save Serper.dev API key in .env as SERPER_API_KEY
# Save OpenAI API key in .env as OPENAI_API_KEY

import os
HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Import chainlit
import chainlit as cl

from crewai import Agent, Crew, Process
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

from tasks import ReadWriteTasks
from agents import ReadWriteAgents

tasks = ReadWriteTasks()
agents = ReadWriteAgents()

topic = 'Best functional health suggestions to cause hormesis'

print("## Welcome to the Research and Write Crew")
print('-------------------------------')
print(topic)

# Create Agents
researcher_agent = agents.research_agent()
writer_agent = agents.writer_agent()

# Create Tasks
researcher_task = tasks.research_task(researcher_agent, topic)
writer_task = tasks.write_task(writer_agent, topic)

# Create Crew responsible for Copy
crew = Crew(
	agents=[
		researcher_agent,
		writer_agent,
	],
	tasks=[
		researcher_task,
		writer_task
	],
  process=Process.sequential, #sequential is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

result = crew.kickoff()


# Print results
print("\n\n################################################")
print("## Here is the result")
print("################################################\n")
print(result)

# Chainlit execution
@cl.on_message
async def main(message: cl.Message):
	query = topic
	response = result
	await cl.Message(response).send()
