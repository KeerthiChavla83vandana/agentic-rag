1. How to Start Containers & View Backend Logs
Option A: Run in background and follow logs (Recommended)
Start all services in detached mode:

bash
docker compose up -d
View live streaming logs of only the backend:

bash
docker compose logs -f wmf_qa_backend
View the last 100 lines and follow live logs:

bash
docker compose logs --tail=100 -f wmf_qa_backend
View logs for all services combined:

bash
docker compose logs -f
(Press Ctrl + C at any time to exit log viewing without stopping the services).

Option B: Run in foreground (See all logs live)
bash
docker compose up
2. How to Test Your Application
Once docker compose up -d is running:

Component	URL	Description
Streamlit UI	http://localhost:8501	Main user interface to upload documents and run RAG QA queries
FastAPI Backend Docs	http://localhost:8001/docs	Interactive Swagger UI to test endpoints directly
Backend Health Check	http://localhost:8001/health	Verify FastAPI backend status
Quick CLI test for backend health:
bash
curl http://localhost:8001/health
3. How to Check PostgreSQL DB via Web UI (pgAdmin 4)
Your docker-compose.yml already includes pgAdmin 4!

Open http://localhost:8080 in your browser.
Login with credentials:
Email: admin@admin.com
Password: admin
Click Add New Server (or right-click Servers $\rightarrow$ Register $\rightarrow$ Server):
General tab:
Name: wmf_qa_db
Connection tab:
Host name/address: wmf_qa_db (internal docker network name)
Port: 5432
Maintenance database: wmf_ip_qa
Username: wmf_user
Password: secure_password
Click Save.
In the left panel, navigate to:
Servers $\rightarrow$ wmf_qa_db $\rightarrow$ Databases $\rightarrow$ wmf_ip_qa $\rightarrow$ Schemas $\rightarrow$ public $\rightarrow$ Tables
Right-click on documents or document_chunks $\rightarrow$ View/Edit Data $\rightarrow$ All Rows to view stored data and column values.
4. How to Check PostgreSQL DB via Command Line (psql)
You can query the database directly from your terminal using docker exec:

Connect to PostgreSQL shell:
bash
docker exec -it wmf_qa_db psql -U wmf_user -d wmf_ip_qa
Handy psql Commands:
Once inside the wmf_ip_qa=# prompt:

List all tables:

sql
\dt
Inspect structure/columns of documents table:

sql
\d documents
Inspect structure/columns of document_chunks table:

sql
\d document_chunks
Check stored documents:

sql
SELECT document_id, source_file, file_path, uploaded_at, embedding_status FROM documents;
Check stored text chunks & metadata:

sql
SELECT chunk_id, document_id, substring(content from 1 for 50) AS snippet, metadata FROM document_chunks LIMIT 5;
Count total records:

sql
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM document_chunks;
Exit psql shell:

sql
\q
Quick Cheat Sheet Summary
bash
# Start services
docker compose up -d
# View backend logs
docker compose logs -f wmf_qa_backend
# Connect to DB CLI
docker exec -it wmf_qa_db psql -U wmf_user -d wmf_ip_qa -c "\dt"
# Stop services
docker compose down




ase 1 Core Architecture (100% Implemented in Code)
Proposal Feature	Proposal Specification	Our Implementation Status
Document Ingestion	Docling layout-aware PDF parsing	✅ Done (app/pipeline/ingestion.py using Docling)
Database & Search	PostgreSQL + pgvector + Full-Text Search	✅ Done (RDS PostgreSQL schema with HNSW & GIN indexes)
Embeddings	Multilingual model	✅ Done (BAAI/bge-m3 1024-dim dense vectors)
Adaptive Query Router	Intent classification	✅ Done (app/pipeline/router.py using NVIDIA NIM API)
Hybrid Retrieval	Dense + Sparse fused via RRF	✅ Done (app/pipeline/retrieval.py RRF fusion)
Corrective Loop	Parallel grading & query rewriting	✅ Done (app/pipeline/grader.py via asyncio.gather)
Cited Synthesis	Citation-enforced LLM synthesis	✅ Done (meta/llama-3.2-11b-vision-instruct on NVIDIA NIM)
Strict Language Matching	Match query language (English/Kannada)	✅ Done (Enforced strictly in prompt rules)
Terminal Telemetry	Step-by-step debug logging	✅ Done (docker compose logs -f wmf_qa_backend)
Frontend UI	Streamlit UI + Execution Tracer	✅ Done (Running on http://localhost:8501)
🟡 Phase 2 / Production Roadmaps (Next Milestones for Discussion)
Text-to-SQL Branch: SQL query generation for structured numerical tables (FPC member counts, budgets).
Cross-Encoder Reranker: Integrating bge-reranker-large for top-K re-ranking before synthesis.
Voice Audio Pipeline: Indic ASR (Sarvam / IndicWhisper) + CPU PII Redaction (Microsoft Presidio) before indexing field audio.

Production Best Practice (Developer Debugger vs. End-User Mode):
In Development / Testing Phase (Current): Showing the Execution Tracer Debugger on the Streamlit frontend is beneficial because it proves to managers and testers that the system is retrieving real data, ranking accurately, and grading chunks correctly.
In Production Phase (Phase 2):
End-Users (Funders / Field Staff) see ONLY the clean Synthesized Answer + Verified Citations.
Developers & Engineers see the detailed debug trace (Chunks, RRF Scores, YES/NO grades) in Terminal Logs / AWS CloudWatch / Langfuse Tracing.



ype 1: Document RAG Questions (Unstructured PDFs & Transcripts)
What it is: Questions looking for qualitative facts, stories, or findings inside uploaded PDFs and audio transcripts.
Example Questions:
"What are the major challenges faced by custodian farmers in Uttara Kannada?"
"What feedback did farmers give about betel vine landrace conservation?"
How Router Handles It: Routes to vector_db $\rightarrow$ Performs vector search on PDFs $\rightarrow$ Returns cited answer.
📌 Type 2: Structured Data Questions (Numbers & Metrics)
What it is: Questions asking for exact numbers, totals, dates, or targets stored in database tables.
Example Questions:
"How many total female farmers registered with the Belagavi FPC in 2025?"
"What is the remaining training budget for the Shivamogga cluster?"
How Router Handles It: Routes to text_to_sql $\rightarrow$ Runs SQL query on PostgreSQL tables $\rightarrow$ Returns exact number/table.
📌 Type 3: Combined / Hybrid Questions (Docs + Database Numbers)
What it is: Questions that need both exact numbers from database tables AND explanations from document reports.
Example Questions:
"Which FPCs missed their 2024 membership target, and what reasons were recorded in their meeting minutes?"
How Router Handles It: Runs SQL for numbers + searches meeting minutes for text $\rightarrow$ Combines both into one report.
📌 Type 4: General Knowledge Questions (No DB Lookup Needed)
What it is: General definition or guidance questions that don't need any local files or database records.
Example Questions:
"What is the legal definition of a Farmer Producer Company (FPC) under Indian Law?"
"How do I conduct an effective field survey with smallholder farmers?"