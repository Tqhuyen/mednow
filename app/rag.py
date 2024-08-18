from pymongo.mongo_client import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_upstage import UpstageEmbeddings
import getpass
import dotenv
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_experimental.text_splitter import SemanticChunker
import urllib.request
from urllib.request import Request, urlopen


load_dotenv()

mongodb_uri  =os.getenv("MONGODB_ATLAS_CLUSTER_URI")
api_key_upstage = os.getenv("UPSTAGE_API_KEY")

os.environ["UPSTAGE_API_KEY"] = api_key_upstage


# Connect to your Atlas cluster
client = MongoClient(mongodb_uri)
# Define collection and index name
DB_NAME = "langchain_db"

ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
COLLECTION_NAME = "medical-data"

db_collection = client[DB_NAME][COLLECTION_NAME]


# Upload data to RAG using Upstage API and MongoDB
def add_pdf_data_to_rag(filename):
    global client, db_collection, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME, DB_NAME, api_key_upstage
    
    layzer = UpstageLayoutAnalysisLoader(file_path=filename, api_key=api_key_upstage, output_type="html", use_ocr=True)
    docs = layzer.load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=1000, chunk_overlap=100, language=Language.HTML
    )

    splits = text_splitter.split_documents(docs)

    unique_splits = [split for split in splits if not db_collection.find_one({"text":split.page_content})]

    if len(unique_splits) > 0:
        MongoDBAtlasVectorSearch.from_documents(
        documents=splits,
        collection=db_collection,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large", api_key=api_key_upstage),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )


def hybrid_search(client, query):
    global db_collection, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME, DB_NAME, api_key_upstage
    vector_penalty = 4
    keyword_penalty = 6
    return client.aggregate(
        [
            {
                # $vectorSearch stage to search the embedding field for the query specified as vector embeddings in the queryVector field of the query.
                # The query specifies a search for up to 100 nearest neighbors and limit the results to 20 documents only. This stage returns the sorted documents from the semantic search in the results.
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": UpstageEmbeddings(
                        model="solar-embedding-1-large", api_key=api_key_upstage
                    ).embed_query(query),
                    "numCandidates": 10,
                    "limit": 5,
                }
            },
            {
                # $group stage to group all the documents in the results from the semantic search in a field named docs.
                "$group": {"_id": None, "docs": {"$push": "$$ROOT"}}
            },
            {
                # $unwind stage to unwind the array of documents in the docs field and store the position of the document in the results array in a field named rank.
                "$unwind": {"path": "$docs", "includeArrayIndex": "rank"}
            },
            {
                # $addFields stage to add a new field named vs_score that contains the reciprocal rank score for each document in the results.
                # Here, reciprocal rank score is calculated by dividing 1.0 by the sum of rank, the vector_penalty weight, and a constant value of 1.
                "$addFields": {
                    "vs_score": {
                        "$divide": [1.0, {"$add": ["$rank", vector_penalty, 1]}]
                    }
                }
            },
            {
                # $project stage to include only the following fields in the results: vs_score, _id, title, text
                "$project": {
                    "vs_score": 1,
                    "_id": "$docs._id",
                    "title": "$docs.title",
                    "text": "$docs.text",
                }
            },
            {
                # $unionWith stage to combine the results from the preceding stages with the results of the following stages in the sub-pipeline
                "$unionWith": {
                    "coll": COLLECTION_NAME,
                    "pipeline": [
                        {
                            # $search stage to search for movies that contain the query in the text field. This stage returns the sorted documents from the keyword search in the results.
                            "$search": {
                                "index": "text",
                                "phrase": {"query": query, "path": "text"},
                            }
                        },
                        {
                            # $limit stage to limit the output to 15 results only.
                            "$limit": 15
                        },
                        {
                            # $group stage to group all the documents from the keyword search in a field named docs.
                            "$group": {"_id": None, "docs": {"$push": "$$ROOT"}}
                        },
                        {
                            # $unwind stage to unwind the array of documents in the docs field and store the position of the document in the results array in a field named rank.
                            "$unwind": {"path": "$docs", "includeArrayIndex": "rank"}
                        },
                        {
                            # $addFields stage to add a new field named kws_score that contains the reciprocal rank score for each document in the results.
                            # Here, reciprocal rank score is calculated by dividing 1.0 by the sum of the value of rank, the full_text penalty weight, and a constant value of 1.
                            "$addFields": {
                                "kws_score": {
                                    "$divide": [
                                        1.0,
                                        {"$add": ["$rank", keyword_penalty, 1]},
                                    ]
                                }
                            }
                        },
                        {
                            # $project stage to include only the following fields in the results: kws_score, _id, title, text
                            "$project": {
                                "kws_score": 1,
                                "_id": "$docs._id",
                                "title": "$docs.title",
                                "text": "$docs.text",
                            }
                        },
                    ],
                }
            },
            {
                # $project stage to include only the following fields in the results: _id, title, text, vs_score, kws_score
                "$project": {
                    "title": 1,
                    "vs_score": {"$ifNull": ["$vs_score", 0]},
                    "kws_score": {"$ifNull": ["$kws_score", 0]},
                    "text": 1,
                }
            },
            {
                # $project stage to add a field named score that contains the sum of vs_score and kws_score to the results.
                "$project": {
                    "score": {"$add": ["$kws_score", "$vs_score"]},
                    "title": 1,
                    "vs_score": 1,
                    "kws_score": 1,
                    "text": 1
                }
            },
            # $sort stage to sort the results by score in descending order.
            {"$sort": {"score": -1}},
            #   $limit stage to limit the output to 10 results only.
            {"$limit": 10},
        ]
    )


def retrive_answer(message):
    global db_collection
    result = hybrid_search(db_collection, f"{message}")
    ans = ""
    for doc in result:
        ans += doc['text']
    return ans
