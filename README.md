# Holistic Wellness AI Suite
 Final project for Codepath AI 6-week bootcamp

 Holistic wellness is a 2 part AI solution:
 1. RAG coach/therapist chatbot that helps users explore their emotions more deeply and understand their inner world
 2. Multi-agentic researcher and report generator to help users understand the latest developments in the health and wellness space, and how to integrate into daily life

## RAG Emotions Chatbot
The dataset used is from English Twitter messages with 6 basic emotions: anger, fear, joy, love, sadness, and surprise (https://huggingface.co/datasets/dair-ai/emotion)
It is deployed on Hugging Face Spaces (https://huggingface.co/spaces/blockchaing/holisticwellness), and additional data from the Feelings Wheel (https://www.calm.com/blog/the-feelings-wheel) is used in the prompt to extend the bot's knowledge base.

Identifying your specific emotions can help you gain a deeper understanding and appreciation for yourself and your inner world, which is why it can be important to name specific emotions. The Feelings Wheels is a tool to help those who have less practice tuning into their emotional landscape.

The code relating to the chatbot is found in app.py


## Multi-agentic Health and Wellness Report Generator
[CrewAI](https://www.crewai.com/) is used to set up a 2 agent system, a researcher agent and a writer agent.
The code relating to this tool is found in main.py, with agents defined in agents.py, and tasks defined in tasks.py
Serper.dev is used by the researcher agent to crawl the web for relevant data
The writer agent then takes over to create an organized, concise, formatted report in markup, in the form of new-routine.md

For the moment, the topic is hardcoded but that could easily be changed
Additionally, it is deployed locally using Chainlit, which displays the report nicely. The file new-routine.md can also be used to view or post the report

## Secrets
Located in the .env file. For this project the following are needed: Hugging Face, OpenAI, Serper.dev

## This project uses:

- [Pandas DataFrames](https://pandas.pydata.org/docs/reference/io.html)
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/)
- [OpenAI Embeddings](https://python.langchain.com/v0.1/docs/integrations/text_embedding/openai/)
- [Facebook AI Similarity Search (FAISS)](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)
- [CrewAI](https://www.crewai.com/)
- [Chainlit](https://chainlit.io/)

## To Run
- Install dependencies ```pip install -r requirements.txt```
- Install chainlit ```pip install chainlit```
- Test chainlit to make sure it's working ```chainlit hello``` This should spawn the Chainlit UI and ask for your name. Quit out of Chainlit if working
- Make sure your secrets are saved to a .env file. The format should be:
  ```python
  HF_TOKEN=<hf_secret>
  OPENAI_API_KEY=<openai_secret>
  SERPER_API_KEY=<serper_secret>
  ```
### To Run the Chatbot    
- Deploy your chatbot to Chainlit ```chainlit run app.py -w```
- You can view and interact with your chatbot at (http://localhost:8000/)
- Or it is permanently deployed on [Hugging Face Spaces](https://huggingface.co/spaces/blockchaing/holisticwellness)

### To Run the Multi-Agent Report
- Deploy your agents to Chainlit ```chainlit run main.py -w```
- You can view the agents working on your terminal, and the report will be ready at (http://localhost:8000/) when complete
- A new-routine.md file will also be saved to your directory
