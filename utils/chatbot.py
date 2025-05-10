import os
from dotenv import load_dotenv
import json
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import re

# Load environment variables
load_dotenv()

class UniversityChatbot:
    def __init__(self):
        """Initialize the University Chatbot with necessary components"""
        # Initialize language model
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Qdrant client for vector search
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Define system prompt templates
        self.system_prompts = {
            "English": (
                "You are a helpful university assistant chatbot. "
                "You provide accurate information about courses, schedules, exams, professors, "
                "and library resources based on the university data. "
                "Be concise, helpful, and friendly. "
                "If you don't know the answer, admit it politely and don't make up information. "
                "Context information is provided below. "
                "Given this information, please answer the user's question.\n\n"
                "Context: {context}\n"
            ),
            "Arabic": (
                "أنت روبوت مساعد جامعي مفيد. "
                "أنت تقدم معلومات دقيقة عن الدورات والجداول والامتحانات والأساتذة "
                "وموارد المكتبة استنادًا إلى بيانات الجامعة. "
                "كن موجزًا ومفيدًا وودودًا. "
                "إذا كنت لا تعرف الإجابة، اعترف بذلك بأدب ولا تختلق المعلومات. "
                "يتم توفير معلومات السياق أدناه. "
                "بناءً على هذه المعلومات، يرجى الإجابة على سؤال المستخدم.\n\n"
                "السياق: {context}\n"
            )
        }
        
        # Define query classifiers for improved retrieval
        self.query_classifiers = {
            "course": r"\b(course|class|syllabus|prerequisite|credit)\b",
            "exam": r"\b(exam|test|assessment|grade|score)\b",
            "faculty": r"\b(professor|instructor|faculty|teacher|lecturer|staff)\b",
            "schedule": r"\b(schedule|timetable|time|date|calendar|when)\b",
            "library": r"\b(library|book|resource|borrow|rent)\b"
        }
        
    def _classify_query(self, query):
        """Classify the query to optimize retrieval"""
        query_type = "general"
        confidence = 0
        
        for category, pattern in self.query_classifiers.items():
            matches = re.findall(pattern, query.lower())
            if len(matches) > confidence:
                query_type = category
                confidence = len(matches)
        
        return query_type, confidence
    
    def _create_metadata_filter(self, query_type):
        """Create metadata filters based on query type"""
        if query_type == "general":
            return None
            
        if query_type == "course":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="course_catalog")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="syllabus")
                    )
                ]
            )
            
        if query_type == "exam":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="exam_schedule")
                    )
                ]
            )
            
        if query_type == "faculty":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="faculty_info")
                    )
                ]
            )
            
        if query_type == "library":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="library_resource")
                    )
                ]
            )
            
        if query_type == "schedule":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.entities.dates", 
                        match=qdrant_models.IsNotEmpty()
                    )
                ]
            )
            
        return None
    
    def _extract_entities_from_query(self, query):
        """Extract key entities from query for better search"""
        entities = {}
        
        # Extract course codes (e.g., CSC 226)
        course_pattern = r'\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b'
        course_matches = re.findall(course_pattern, query)
        if course_matches:
            entities["course_codes"] = course_matches
            
        # Extract names (simple heuristic, not ideal)
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        name_matches = re.findall(name_pattern, query)
        if name_matches:
            entities["names"] = name_matches
            
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        date_matches = re.findall(date_pattern, query)
        if date_matches:
            entities["dates"] = date_matches
            
        return entities
        
    def _retrieve_context(self, query, top_k=5):
        """
        Retrieve relevant context from the vector database using enhanced metadata
        """
        try:
            # Generate embeddings for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Classify query
            query_type, confidence = self._classify_query(query)
            
            # Extract entities from query
            entities = self._extract_entities_from_query(query)
            
            # Create metadata filter
            metadata_filter = self._create_metadata_filter(query_type)
            
            # If we have a course code, add a specific filter for it
            if "course_codes" in entities and entities["course_codes"]:
                # Course codes might be in different formats, try to match flexibly
                course_code_condition = qdrant_models.Filter(
                    should=[
                        qdrant_models.FieldCondition(
                            key="metadata.entities.course_codes", 
                            match=qdrant_models.MatchAny(any=entities["course_codes"])
                        ),
                        qdrant_models.FieldCondition(
                            key="metadata.chunk_entities.course_codes", 
                            match=qdrant_models.MatchAny(any=entities["course_codes"])
                        )
                    ]
                )
                
                if metadata_filter:
                    # Combine the filters with AND logic
                    metadata_filter = qdrant_models.Filter(
                        must=[metadata_filter, course_code_condition]
                    )
                else:
                    metadata_filter = course_code_condition
                    
            # Similar approach for other entity types...
            
            # Search in Qdrant with filter if applicable
            if metadata_filter:
                search_results = self.qdrant_client.search(
                    collection_name="university_data",
                    query_vector=query_embedding,
                    query_filter=metadata_filter,
                    limit=top_k
                )
            else:
                search_results = self.qdrant_client.search(
                    collection_name="university_data",
                    query_vector=query_embedding,
                    limit=top_k
                )
            
            # Extract and format the results with metadata context
            contexts = []
            for hit in search_results:
                text = hit.payload.get("text", "")
                metadata = hit.payload.get("metadata", {})
                
                # Add source information
                source_info = ""
                if metadata.get("structure", {}).get("doc_title"):
                    source_info += f"Document: {metadata['structure']['doc_title']}\n"
                if metadata.get("structure", {}).get("heading"):
                    source_info += f"Section: {metadata['structure']['heading']}\n"
                    
                # Format entity information if relevant
                entity_info = ""
                course_codes = metadata.get("chunk_entities", {}).get("course_codes", [])
                if course_codes:
                    entity_info += f"Course codes: {', '.join(str(code) for code in course_codes[:3])}\n"
                    
                # Combine the information
                context_item = f"{source_info}{entity_info}\n{text}"
                contexts.append(context_item)
            
            return "\n\n---\n\n".join(contexts)
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def get_response(self, query, language="English"):
        """
        Generate a response to a user query using enhanced RAG
        """
        try:
            # Retrieve relevant context
            context = self._retrieve_context(query)
            
            # If no context found, use a placeholder
            if not context:
                context = "No specific information available."
            
            # Format the system message with context
            system_content = self.system_prompts[language].format(context=context)
            
            # Generate response using langchain
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            response = self.llm.generate([messages])
            return response.generations[0][0].text
        except Exception as e:
            print(f"Error generating response: {e}")
            if language == "English":
                return "I'm sorry, I encountered an error while processing your request. Please try again later."
            else:
                return "آسف، لقد واجهت خطأ أثناء معالجة طلبك. الرجاء المحاولة مرة أخرى لاحقًا." 