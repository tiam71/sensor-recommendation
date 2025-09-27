from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(model_name)
model.save("/Users/laihy/Desktop/智能產品推薦與自動報價系統/testfront")
