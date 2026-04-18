# RAG
Using Rag to extract context based on query from a pdf


1- Create a hugging face api key. Otherwise, it won't be able to download the trained llm. Look for RAGPipeLine.py class. Replase HF_KEY = "YOUR_TOKEN" with your token. 
2- load_pdf(self, pdf_path) function is where I have parsed the Quran the way I need it. You gotta modify according to your need. 
3- In the main.py, I loaded the pdf directly from the root directory of the project. document = "quran-english.pdf". chande your pdf or write your own method to connect with jenkins to retrieve the data.
4- Since my target was to get a precise answer, my new_max_token = 2048. It takes 3-5 minutes to answer your one question. But if you want small answer, then just change the max_new_token to 128 or 256 or something like that
5- Since my GPU does not support that large data, I have only did it with CPU. That's one of the reason it takes a while to generate the answers.
