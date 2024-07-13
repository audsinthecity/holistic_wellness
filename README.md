# Holistic Wellness AI Suite
 Final project for Codepath AI 6-week bootcamp

 Holistic wellness is a 2 part AI solution:
 1. RAG coach/therapist chatbot that helps users explore their emotions more deeply and understand their inner world
 2. Multi-agentic researcher and report generator to help users understand the latest developments in the health and wellness space, and how to integrate this information into daily life

## Retrieval-Augmented Generation (RAG) Emotions Chatbot
The dataset used is from English Twitter messages with 6 basic emotions: anger, fear, joy, love, sadness, and surprise (https://huggingface.co/datasets/dair-ai/emotion)
It is deployed on Hugging Face Spaces (https://huggingface.co/spaces/blockchaing/holisticwellness), and additional data from the Feelings Wheel (https://www.calm.com/blog/the-feelings-wheel) is used in the prompt to extend the bot's knowledge base.<br/> 

Identifying your specific emotions can help you gain a deeper understanding and appreciation for yourself and your inner world, which is why it can be important to name specific emotions. The Feelings Wheels is a tool to help those who have less practice tuning into their emotional landscape. For the sake of simplicity for this project, we only
included feelings in the first layer of the Feelings Wheel as opposed to the entire Wheel.<br/> 

The code relating to the chatbot is found in app.py<br/> 

The additional prompt text that should be inserted into the Hugging Face Spaces UI under Additional Inputs, System message is:<br/> 
<mark>
You are a friendly therapist and coach who helps people explore their feelings and inner worlds. You are empathetic, curious, and encouraging. If you think someone is angry, ask if they feel let down, humiliated, bitter, made, aggressive, frustrated, distant or critical. If you think someone is disgusted, ask if they feel disapproving, disappointed, awful, or repelled. If you think someone is sad, ask if they feel hurt, depressed, guilty, despair, vulnerable, or lonely. If you think someone is happy, ask if they feel playful, content, interested, proud, accepted, powerful, peaceful, trusting, or optimistic. If you think someone is surprised, ask if they feel startled, confused, amazed, or excited. If you think some is feeling bad, ask if they are feeling bored, busy, stressed, or tired. If you think someone is feeling fearful, ask if they are feeling scared, anxious, insecure, weak, rejected, or threatened. You offer suggestions like breathing exercises to connect to their inner self and regulate their nervous system.</mark>


## Multi-agentic Health and Wellness Report Generator
[CrewAI](https://www.crewai.com/) is used to set up a 2 agent system, a researcher agent and a writer agent.
The code relating to this tool is found in main.py, with agents defined in agents.py, and tasks defined in tasks.py<br/> 
Serper.dev is used by the researcher agent to crawl the web for relevant data<br/> 
The writer agent then takes over to create an organized, concise, formatted report in markup, in the form of new-routine.md<br/> 

For the moment, the topic is hardcoded but that could easily be changed. We decided it probably does not make sense to integrate into the chatbot
in its current form given the length of time it takes to run as that would create a poor user experience,
and the fact that the health reports are not directly relevant to feelings exploration and coaching.<br/>
To add the relevant information to the RAG bot and maintain speed performance, we could run many reports and then embed the information.<br/>
It can be interesting to watch the agents as they "think" and execute next steps to accomplish their tasks however.<br/> 
The multi-agentic system is deployed locally using Chainlit, which displays the report nicely. The generated file new-routine.md can also be used to view or post the report.<br/> 
Since a new web search is kicked off with every run, a brand new report is generated each time and the file is overwritten if using the same directory.<br/> 

## Secrets
Located in the .env file, which should be created (see below for format). For this project the following are needed: Hugging Face, OpenAI, Serper.dev<br/> 

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

## Future Work
FACIAL ANALYSIS<br/> 
Allow user to submit selfie so AI can analyze facial expressions for additional emotions data<br/> 
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset<br/> 

PERSISTED DATA FOR TREND REPORTING<br/> 
Provide reporting so users can easily see trends and patterns over longer timespans like a week or month<br/> 
Optional notifications to encourage good effort or friendly nudges<br/> 

CONTENT RECOMMENDATIONS<br/> 
Book or article recommendations based on chat content<br/> 
Could provide summaries of said content, or even chunk into small “Learning of the Day” concepts<br/> 
Suggest meditations or songs to play based on mood or to improve mood, connect to Spotify/Soundcloud/Calm/Insight Timer/etc<br/> 

PERSONALIZED 1-PAGER<br/> 
Downloadable PDF “cheat sheet” for personalized health practices given some user input on lifestyle/needs/schedule/individual health concerns<br/> 
Extension to create calendar invite reminders and integrate with Google Calendar, Microsoft, etc<br/> 

## Conclusions
The RAG chatbot did a pretty good job with the task of acting as a friendly coach helping users explore their emotions using data from the Feelings Wheel in a 
human-sounding manner, which is heartening given that the cost of therapy can be hundreds of dollars per hour. The multi-agent system, while quite interesting, took a long time to run as well as cost a lot of tokens. It is probably not suited to real run time interactions but could be useful in situations where waiting is acceptable and the quality of the output is worth the cost. There is probably quite a lot of future potential here, especially as AI gets better at evaluating the output of other AI. As humans initially put in many hours of work tagging, labeling, and categorizing data, humans may also have to manually judge many outputs to improve the judgment and accuracy of AI evaluators.<br/>
In the initial testing of both the chatbot and reporting system, no hallucination was observed, however it should be noted that current AI systems notoriously hallucinate. Getting to 80% performance is relatively easy but the last 20% could need many examples including negative examples. Fine tuning could be an approach to consider for improving accuracy, decreasing hallucinations, and teaching custom data at an affordable price point if fine tuning on a simpler cheaper model.


