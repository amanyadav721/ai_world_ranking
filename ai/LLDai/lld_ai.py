import json
import os 
import io
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def lld_creator(data:dict) -> json:
    """
    Creates a Low-Level Design (LLD) document using the Groq LLM.

    Args:
        text (str): The text to summarize.
    
    Returns:
        str: The summarized text.

    """


    system_prompt = f"""
    You are a senior software architect with 15+ years of experience in designing scalable, fault-tolerant, and optimized systems across startups, enterprise projects, and large-scale products.

    Given the following input:

    Category: {{category}}  // e.g., business, startup, project  
    Scale: {{scale}}        // e.g., small, medium, large  
    Type of Product: {{type}} // e.g., brand, tech, e-commerce, SaaS  
    Description: {{description}} // detailed description of the product  

    Here is the data provided by the user:
    {json.dumps(data)}

    our goal is to understand the product deeply and produce a **complete, production-grade Low-Level Design (LLD)** in **strict JSON format**, adhering to **industry-level standards**, focusing on **scalability, performance, fault tolerance, maintainability, and modularity**.

    Include explanations for **why** each key choice was made at the end of the design.


                ### Output Format (Strict JSON):

                {{
            "techStack_and_architecture": {{
                "language": "",
                "framework": "",
                "database": "",
                "architecture_type": "monolith | microservices",
                "justification": "Why this architecture and stack were chosen based on scale, type, and needs"
            }},
            "backend_lld": {{
                "services": [
                {{
                    "name": "",
                    "description": "",
                    "api_contracts": [
                    {{
                        "method": "GET | POST | PUT | DELETE",
                        "path": "",
                        "request": {{ }},
                        "response": {{ }}
                    }}
                    ],
                    "db_schema": {{
                    "tables": [
                        {{
                        "name": "",
                        "columns": [
                            {{
                            "name": "",
                            "type": "",
                            "constraints": ""
                            }}
                        ],
                        "relationships": []
                        }}
                    ]
                    }}
                }}
                ]
            }},
            "frontend_lld": {{
                "page_flow": [
                "Page 1",
                "Page 2",
                ...
                ],
                "components": [
                {{
                    "name": "",
                    "children": [],
                    "state_management": "Context API | Redux | Zustand | Signals",
                    "reasoning": "Why this state management method fits the component/page complexity"
                }}
                ]
            }},
            "integrations": {{
                "payment_gateway": {{
                "provider": "",
                "integration_method": ""
                }},
                "communication": {{ 
                "email_service": "",
                "sms_service": ""
                }},
                "analytics": {{
                "tool": "",
                "events_tracked": []
                }}
            }},
            "Deployment": {{
                "cloud_architecture": {{        
                "provider": "AWS | GCP | Azure",
                "diagram_description": "Describe infrastructure components like Load Balancer, DB, CDN, VPC, S3, etc."
                }},
                "ci_cd_pipeline": {{    
                "tools": [
                    "GitHub Actions | GitLab CI | Jenkins"
                ],
                "stages": [
                    "build",
                    "test",
                    "lint",
                    "deploy"
                ]
                }}
                    }},
            "Design_Explanation": {{
                "overall_strategy": "Explain the design philosophy",
                "scalability_rationale": "How it handles growth",
                "fault_tolerance_measures": "How failures are isolated and recovered",
                "performance_considerations": "Caching, load balancing, DB optimizations",
                "future_proofing": "How the system can evolve over time"
            }}    
            }}

    Strict Guidelines:
                    1. No text outside the JSON object.

                    2. No markdown or code blocks in the JSON object.

                    3. Must be valid JSON open and close braces.

                    4. Structure must be exactly as specified, with no extra fields or changes.

                    5. Do not add ''' or '''json to the JSON object before or after the JSON object.

                    7. You reponse should start with open brace and end with close brace.
    """

    
    messages = [
    ("system", system_prompt),
    ("human", json.dumps(data)),  # Convert list/dict to a JSON string
  ]

    ai_msg = llm.invoke(messages)
    response = json.loads(ai_msg.content)
    return response