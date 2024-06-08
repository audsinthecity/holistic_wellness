from textwrap import dedent
from crewai import Agent

from tools.ExaSearchTool import ExaSearchTool

class ReadAndWriteAgents():
	def research_agent(self):
		return Agent(
			role='Research Specialist',
			goal='Uncover best practices and the latest scientific research in {topic}',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					As a Research Specialist, your mission is to explore and share knowledge that could help humanity thrive holistically in all
          areas of health and wellness. Your insights
					will lay the groundwork for a personalized health plan."""),
			verbose=True,
      memory=True,
      allow_delegation=True
		)

	def writer_agent(self):
		return Agent(
			role='Writer',
			goal='Analyze the current industry trends, challenges, and actionable insights relevant to {topic}',
			tools=ExaSearchTool.tools(),
			backstory=dedent("""\
					With a flair for finding the latest health and wellness trends, your analysis will identify key trends,
					new research, and actionable steps
					for how users can incorporate this knowledge into their daily life."""),
			verbose=True,
      memory=True,
      allow_delegation=False
		)

