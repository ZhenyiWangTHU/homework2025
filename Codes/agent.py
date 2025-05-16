from typing import List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import json
import os
from datetime import datetime
import glob


os.environ["OPENAI_API_KEY"] = ''



METABOLITE_TEMPLATE = """You are a professional bioinformatics expert. Please generate a structured description for the following metabolite.
The description should include the following aspects:
1. Chemical Structure
2. Biological Function
3. Metabolic Pathway
4. Related Diseases
5. Research Significance

Metabolite Name: {metabolite_name}

Please return in JSON format with the following fields:
{{
    "chemical_structure": "Description of chemical structure",
    "biological_function": "Description of biological function",
    "metabolic_pathway": "Description of metabolic pathway",
    "related_diseases": "Description of related diseases",
    "research_significance": "Description of research significance"
}}

IMPORTANT: Return ONLY the JSON object, nothing else.
"""

class MetaboliteProcessor:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.checkpoint_dir = checkpoint_dir
        self.descriptions = {}
        self.embeddings = {}
        self.processed_count = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def process_single_metabolite(self, metabolite_name: str) -> bool:
        try:
            prompt = ChatPromptTemplate.from_template(METABOLITE_TEMPLATE)
            chain = prompt | self.llm
            response = chain.invoke({"metabolite_name": metabolite_name})
            
            print(f"\nProcessing: {metabolite_name}")

            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            description = json.loads(content)
            self.descriptions[metabolite_name] = description
            
            description_text = json.dumps(description, ensure_ascii=False)
            embedding = self.embeddings_model.embed_query(description_text)
            self.embeddings[metabolite_name] = embedding
            
            self.processed_count += 1
            print(f"Successfully processed {metabolite_name}")
            
            if self.processed_count % 10 == 0:
                self.save_checkpoint()
            
            return True
            
        except Exception as e:
            print(f"Error processing {metabolite_name}: {str(e)}")
            return False

    def save_checkpoint(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.checkpoint_dir}/checkpoint_{timestamp}.json"
        
        checkpoint_data = {
            "descriptions": self.descriptions,
            "embeddings": self.embeddings,
            "processed_count": self.processed_count
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nCheckpoint saved: {filename}")
        print(f"Total metabolites processed: {self.processed_count}")

    def save_final_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"metabolite_results_{timestamp}.json"
        
        with open(final_filename, "w", encoding="utf-8") as f:
            json.dump({
                "descriptions": self.descriptions,
                "embeddings": self.embeddings
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nFinal results saved to: {final_filename}")
        print(f"Total metabolites processed: {self.processed_count}")

    @classmethod
    def load_latest_checkpoint(cls, checkpoint_dir: str = "checkpoints") -> 'MetaboliteProcessor':
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
        if not checkpoint_files:
            return cls(checkpoint_dir)
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        processor = cls(checkpoint_dir)
        
        with open(latest_checkpoint, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        processor.descriptions = checkpoint_data["descriptions"]
        processor.embeddings = checkpoint_data["embeddings"]
        processor.processed_count = checkpoint_data["processed_count"]
        
        print(f"Loaded checkpoint with {processor.processed_count} processed metabolites")
        return processor

def process_metabolites(metabolite_names: List[str], batch_size: int = 10, resume: bool = True):
    if resume:
        processor = MetaboliteProcessor.load_latest_checkpoint()
        processed_metabolites = set(processor.descriptions.keys())
        remaining_metabolites = [m for m in metabolite_names if m not in processed_metabolites]
        print(f"Resuming processing. {len(processed_metabolites)} metabolites already processed.")
        print(f"Remaining metabolites to process: {len(remaining_metabolites)}")
    else:
        processor = MetaboliteProcessor()
        remaining_metabolites = metabolite_names
    
    print(f"Starting to process {len(remaining_metabolites)} metabolites...")
    
    for i, metabolite in enumerate(remaining_metabolites, 1):
        success = processor.process_single_metabolite(metabolite)
        
        if not success:
            print(f"Failed to process {metabolite}, saving checkpoint and stopping...")
            processor.save_checkpoint()
            break
        
        if i % batch_size == 0:
            print(f"\nCompleted batch of {batch_size} metabolites")
            processor.save_checkpoint()
    
    processor.save_final_results()
    
    return {
        "descriptions": processor.descriptions,
        "embeddings": processor.embeddings
    }