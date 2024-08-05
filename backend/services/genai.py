from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
import logging
import json
import re
from tqdm import tqdm
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from itertools import chain

#Configure log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

class JSONObject(BaseModel):
    concept: str = Field(description="The concept of the subject")
    definition: str = Field(description="The definition of the subject")

class GeminiProcessor:
    def __init__(self,model_name,project):
        self.model = VertexAI(model_name=model_name,project=project)

    def generate_document_summary(self, documents : list, **args):

        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(llm=self.model,chain_type=chain_type,**args)

        return chain.run(documents)
    
    def count_total_tokens(self,docs : list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total tokens...")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total
    
    def get_model(self):
        return self.model
    

class YoutubeProcessor:

    def __init__(self, genai_processor : GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        self.GeminiProcessor = genai_processor
    
    def retrieve_youtube_documents(self,url:str,verbose=False):
        loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)

        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.GeminiProcessor.count_total_tokens(result)

        if verbose:
            logger.info(f"{author}\n{length}\n{title}\n{total_size}\n{total_billable_characters}")

        return result
    
    def find_key_concepts(self,documents : list, sample_size :  int=0, verbose= False):
        #iterate through all the documents of group size N and find the key concepts
        if sample_size > len(documents):
            raise ValueError("Group size is larger than the number of documents")
        
        #Optimize sample_size given no input
        if sample_size == 0:
            sample_size = len(documents) // 5
            if verbose:
                logging.info(f"No sample size specified. Setting number of documents per sample as 5. Sample size: {sample_size}")

        #Find the number of documents in each group
        num_docs_per_group = len(documents) // sample_size + (len(documents) % sample_size > 0)

        #Check thresholds for response quality
        if num_docs_per_group >  10:
            raise ValueError("Each group has more than 10 documents and output quality will be degraded significantly. Increase the sample_size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded. Consider increasing the sample size.")

        #Split the documents in chunk of size num_docs_per_group
        groups = [documents[i : i + num_docs_per_group] for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        batch_cost = 0

        parser = JsonOutputParser(pydantic_object=JSONObject)

        example_template = '''
        Output:

            [
                {
                    "concept": "Large Language Models (LLMs)",
                    "definition": "Powerful AI tools trained on massive datasets to perform tasks like text generation, translation, and question answering."
                },
                {
                    "concept": "Pre-trained and fine-tuned",
                    "definition": "LLMs learn general knowledge from large datasets and specialize in specific tasks through additional training."
                },
                {
                    "concept": "Prompt design",
                    "definition": "Effective prompts are crucial for eliciting desired responses from LLMs."
                },
                {
                    "concept": "Domain knowledge",
                    "definition": "Understanding the specific domain is essential for building and tuning LLMs."
                },
                {
                    "concept": "Parameter-efficient tuning methods",
                    "definition": "This method allows for efficient customization of LLMs without altering the entire model."
                },
                {
                    "concept": "Vertex AI",
                    "definition": "Provides tools for building, tuning, and deploying LLMs for specific tasks."
                },
                {
                    "concept": "Generative AI App Builder and PaLM API",
                    "definition": "Tools for developers to build AI apps and experiment with LLMs."
                },
                {
                    "concept": "Model management tools",
                    "definition": "Tools for training, deploying, and monitoring ML models."
                }
            ]
'''

        logger.info("Finding key concepts...")
        for group in tqdm(groups):
            #Combine content of document per group
            group_content = ''

            for doc in group:
                group_content += doc.page_content

            #Prompt for finding concepts
            prompt = PromptTemplate(
                template= '''
                Find the key concepts and their definitions from the following text:
                {text}.
                Respond only in clean JSON string format without any labels or additional text. The output should look exactly like this
                {examples}

                {format_instructions}
                Respond only according to the format instructions. The examples included are best responses noted by an input and output example.
                ''',
                input_variables=['text'],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            #create a chain
            lang_chain = prompt | self.GeminiProcessor.model | parser

            #run chain
            output_concept = lang_chain.invoke({"text" : group_content, "examples" : example_template})

            # output_concept = output_concept.replace("```json", "").replace("```", "").strip()

            # Regex to remove backticks if generated
            # pattern_start = r"^```json\n"
            # pattern_end = r"\n```$"
            # if re.search(pattern_start, output_concept) and re.search(pattern_end, output_concept):
            #     batch_concepts.append(re.sub(pattern_end, "", re.sub(pattern_start, "", output_concept)))
            # else:  
            batch_concepts.append(output_concept)

            #Post processing observation
            if verbose:
                total_inp_char = len(group_content)
                total_input_cost = (total_inp_char/1000) * 0.000125

                logger.info(f"Running chain on {len(group)} documents")
                logger.info(f"Total input characers: {total_inp_char}")
                logger.info(f"Total input cost: {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375
                logger.info(f"Total output characers: {total_output_char}")
                logger.info(f"Total output cost: {total_output_cost}")

                batch_cost += total_input_cost + total_output_cost
                logger.info(f"Total group cost: {total_input_cost + total_output_cost}")
            
        
        #convert each JSON string in batch_concepts to a Python Dict
        # processed_concepts = [json.loads(concept)  for concept in batch_concepts ]

        logging.info(f"Total Analysis Cost: ${batch_cost}")

        result = list(chain.from_iterable(batch_concepts))

        return result