from textwrap import dedent
from crewai import Task

class ReadWriteTasks():
  # Research task
	def research_task(self, agent, topic):
		return Task(
			description=dedent(f"""\
				Identify the next big trends in {topic}.
				Focus on identifying pros and cons and the overall narrative.
				Your final report should clearly articulate the key points,
				potential benefits to human health, and potential risks.
        """),
			expected_output=dedent("""\
				A detailed report summarizing key findings
				on the latest health and wellness trends."""),
			async_execution=True,
			agent=agent
		)

  # Writing task with language model configuration
	def write_task(self, agent, topic):
		return Task(
			description=dedent("""\
				Compose an insightful article on {topic}.
				Focus on how people can easily incorporate related healthy habits into their daily routines.
				This article should be easy to understand, engaging, and positive.
				The format should be easy to read, including topics highlighted and bulleted lists.
				}"""),
			expected_output=dedent("""\
				An insightful analysis that identifies major trends, potential benefits and
				challenges, and actionable suggestions on {topic} formatted as markdown."""),
			async_execution=False,
			agent=agent,
      		output_file='new-routine.md' # Example of output customization
			)
