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
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

class UniversityChatbot:
    def __init__(self):
        """Initialize the University Chatbot with necessary components"""
        # Initialize language model
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini ",
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
            "library": r"\b(library|book|resource|borrow|rent)\b",
            "admission": r"\b(admission|apply|application|enroll|registration)\b",
            "financial": r"\b(tuition|fee|cost|price|scholarship|financial aid)\b"
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
                        match=qdrant_models.MatchValue(value="course_document")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["course", "syllabus", "prerequisites"])
                    )
                ]
            )
            
        if query_type == "exam":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["exam", "test", "assessment", "grade"])
                    )
                ]
            )
            
        if query_type == "faculty":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.type", 
                        match=qdrant_models.MatchValue(value="faculty_info")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.heading", 
                        match=qdrant_models.MatchText(text="faculty")
                    )
                ]
            )
            
        if query_type == "library":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["library", "resources", "book"])
                    )
                ]
            )
            
        if query_type == "schedule":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["schedule", "timetable", "calendar"])
                    )
                ]
            )
            
        if query_type == "admission":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.heading", 
                        match=qdrant_models.MatchText(text="admission")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["admission", "application", "enroll", "apply"])
                    )
                ]
            )
            
        if query_type == "financial":
            return qdrant_models.Filter(
                should=[
                    qdrant_models.FieldCondition(
                        key="metadata.heading", 
                        match=qdrant_models.MatchText(text="tuition")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.heading", 
                        match=qdrant_models.MatchText(text="scholarship")
                    ),
                    qdrant_models.FieldCondition(
                        key="metadata.keywords", 
                        match=qdrant_models.MatchAny(any=["tuition", "fee", "cost", "scholarship", "financial aid"])
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
            entities["course_codes"] = [f"{code[0]} {code[1]}" for code in course_matches]
            
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
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract potential keywords from the query"""
        # List of common stopwords to filter out
        stopwords = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                    "in", "on", "at", "to", "for", "with", "by", "about", "like", "through",
                    "over", "before", "after", "between", "under", "above", "of", "and", "or",
                    "how", "what", "when", "where", "why", "who", "whom", "which", "there", "that"]
        
        # Convert to lowercase and tokenize by word boundaries
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stopwords and short words
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Look for specific academic terms
        academic_terms = ["admission", "tuition", "scholarship", "credit", "program", 
                         "degree", "major", "minor", "course", "class", "faculty", 
                         "semester", "exam", "library", "deadline", "graduate"]
        
        # Prioritize academic terms if found
        found_terms = [term for term in academic_terms if term in query.lower()]
        
        # Combine unique terms
        all_keywords = list(set(found_terms + keywords))
        
        return all_keywords
        
    def _hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity with keyword and heading matching
        """
        try:
            # 1. Vector similarity search
            query_embedding = self.embeddings.embed_query(query)
            
            # 2. Classify query and create metadata filter
            query_type, confidence = self._classify_query(query)
            metadata_filter = self._create_metadata_filter(query_type)
            
            # 3. Extract entities and keywords
            entities = self._extract_entities_from_query(query)
            keywords = self._extract_keywords_from_query(query)
            
            # 4. Perform vector search (with filter if available)
            if metadata_filter:
                vector_results = self.qdrant_client.search(
                    collection_name="university_data",
                    query_vector=query_embedding,
                    query_filter=metadata_filter,
                    limit=top_k * 2  # Get more results for re-ranking
                )
            else:
                vector_results = self.qdrant_client.search(
                    collection_name="university_data",
                    query_vector=query_embedding,
                    limit=top_k * 2  # Get more results for re-ranking
                )
            
            # 5. Keyword-based search if we have specific entities or keywords
            if keywords or entities:
                # Create a keyword filter for heading and text matching
                keyword_conditions = []
                
                # Add keyword conditions for text content
                for keyword in keywords:
                    if len(keyword) >= 4:  # Only use longer keywords for text search
                        keyword_conditions.append(
                            qdrant_models.FieldCondition(
                                key="text",
                                match=qdrant_models.MatchText(text=keyword)
                            )
                        )
                
                # Add heading match conditions
                for keyword in keywords:
                    if len(keyword) >= 3:  # Shorter words can be okay for headings
                        keyword_conditions.append(
                            qdrant_models.FieldCondition(
                                key="metadata.heading",
                                match=qdrant_models.MatchText(text=keyword)
                            )
                        )
                
                # Add entity conditions
                if "course_codes" in entities:
                    for course_code in entities["course_codes"]:
                        keyword_conditions.append(
                            qdrant_models.FieldCondition(
                                key="text",
                                match=qdrant_models.MatchText(text=course_code)
                            )
                        )
                
                if keyword_conditions:
                    keyword_filter = qdrant_models.Filter(
                        should=keyword_conditions
                    )
                    
                    # Perform keyword search
                    keyword_results = self.qdrant_client.search(
                        collection_name="university_data",
                        query_vector=query_embedding,  # Still use embedding for ordering
                        query_filter=keyword_filter,
                        limit=top_k  # Get top_k results
                    )
                else:
                    keyword_results = []
            else:
                keyword_results = []
            
            # 6. Combine and re-rank results
            combined_results = self._combine_and_rerank_results(
                query, vector_results, keyword_results, keywords, entities
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fall back to basic vector search if hybrid search fails
            try:
                fallback_results = self.qdrant_client.search(
                    collection_name="university_data",
                    query_vector=query_embedding,
                    limit=top_k
                )
                return fallback_results
            except:
                return []
    
    def _combine_and_rerank_results(
        self, 
        query: str, 
        vector_results: List, 
        keyword_results: List, 
        keywords: List[str], 
        entities: Dict[str, List[str]]
    ) -> List:
        """Combine vector and keyword results and rerank based on relevance"""
        # Create a dictionary to track unique results by ID
        unique_results = {}
        
        # Process vector results first (these already have a similarity score)
        for result in vector_results:
            result_id = result.id
            if result_id not in unique_results:
                # Add basic result with the vector score
                unique_results[result_id] = {
                    "id": result_id,
                    "payload": result.payload,
                    "score": {
                        "vector": result.score,
                        "keyword": 0,
                        "heading": 0,
                        "entity": 0,
                        "combined": result.score * 0.7  # Vector gets 70% weight initially
                    }
                }
        
        # Process keyword results and update scores
        for result in keyword_results:
            result_id = result.id
            if result_id in unique_results:
                # Update existing result with keyword score
                unique_results[result_id]["score"]["keyword"] = result.score
                # Recalculate combined score
                unique_results[result_id]["score"]["combined"] += result.score * 0.3
            else:
                # Add new result with the keyword score
                unique_results[result_id] = {
                    "id": result_id,
                    "payload": result.payload,
                    "score": {
                        "vector": 0,
                        "keyword": result.score,
                        "heading": 0,
                        "entity": 0,
                        "combined": result.score * 0.3  # Keyword gets 30% weight initially
                    }
                }
        
        # Further analyze results for heading and entity matches
        for result_id, result in unique_results.items():
            text = result["payload"].get("text", "")
            metadata = result["payload"].get("metadata", {})
            heading = metadata.get("heading", "")
            
            # Check for heading matches
            heading_score = 0
            if heading:
                for keyword in keywords:
                    if keyword.lower() in heading.lower():
                        heading_score += 0.2  # Boost score for each keyword in heading
            
            # Check for entity matches
            entity_score = 0
            for entity_type, entity_values in entities.items():
                for entity in entity_values:
                    if entity.lower() in text.lower():
                        entity_score += 0.3  # Substantial boost for entity matches
            
            # Update scores
            result["score"]["heading"] = heading_score
            result["score"]["entity"] = entity_score
            
            # Recalculate combined score
            result["score"]["combined"] += heading_score + entity_score
        
        # Convert the dictionary to a list and sort by combined score
        result_list = list(unique_results.values())
        result_list.sort(key=lambda x: x["score"]["combined"], reverse=True)
        
        return result_list
    
    def _retrieve_context(self, query, top_k=5):
        """
        Retrieve relevant context from the vector database using hybrid search
        """
        try:
            # Perform hybrid search
            search_results = self._hybrid_search(query, top_k)
            
            # Extract and format the results with metadata context
            contexts = []
            for hit in search_results:
                text = hit["payload"].get("text", "")
                metadata = hit["payload"].get("metadata", {})
                
                # Add source information
                source_info = ""
                if metadata.get("heading"):
                    source_info += f"Section: {metadata['heading']}\n"
                elif metadata.get("doc_title"):
                    source_info += f"Document: {metadata['doc_title']}\n"
                
                # Format entity information if relevant
                entity_info = ""
                if metadata.get("keywords"):
                    keywords = metadata.get("keywords", [])
                    if keywords and isinstance(keywords, list) and len(keywords) > 0:
                        entity_info += f"Keywords: {', '.join(keywords[:3])}\n"
                    
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