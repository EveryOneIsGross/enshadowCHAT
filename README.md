# enshadowCHAT

enshadowCHAT is a 2x2 matrix of instanced chatbots that generates a mesh of slightly different answers and then conditions the worst response, into a high temp stream of unconscious to consider along side most relevent response.
enshadowCHAT is a Python-based conversational AI framework, featuring an ensemble of language models and incorporating a 'shadow' methodology to generate holistic summaries from given prompts. This project leans on the power of GPT-3 and Sentence Transformers to understand and explore prompts deeply and from various perspectives. Emulating how a nn trains data, this framework conditions responses using temperature variance. 

![enshadowCHAT](https://github.com/EveryOneIsGross/enshadowCHAT/assets/23621140/4d2ecc62-2c9c-4852-8c34-7d5ce66ff257)

## Key Features

**Ensemble of Chatbots:**
Utilizes an ensemble-like grid of GPT-3 instances, each independently generating responses to a given prompt. The ensemble mechanism improves response diversity and robustness.

**Shadow Methodology:**
Incorporates a 'shadow' response mechanism which probes the more profound, unconscious level interpretations of prompts, thus providing a multidimensional understanding.

**Holistic Summarization:**
Ranks the responses from chatbots and creates a holistic summary of the best and the shadow responses.

**Semantic Similarity:**
Uses Sentence Transformers to rank responses based on semantic similarity to the prompt.

**Sentiment Analysis:**
Utilizes sentiment analysis to tune the response temperature for subsequent prompts.
