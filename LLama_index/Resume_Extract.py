import os
from dotenv import load_dotenv , find_dotenv
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")



data = open("Resume.pdf", "r")
data

documents = SimpleDirectoryReader("data").load_data()
documents

print(dir(documents[0]))

for document in documents:
    print(document.text)
    print("\n---\n")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") # bge-base embedding model
Settings
Settings.embed_model

# ollama
#Settings.llm = Ollama(model="llama2",request_timeout=600, json_mode=True)
print(dir(documents[0]))


llm = Gemini(api_key= API_KEY , model = 'models/gemini-pro')
llm

index = VectorStoreIndex.from_documents(
    documents,
)

name_tmpl = (
      "Extract the full name of the applicant from the resume. Provide the name in the format:\n"
      "Name: [Full Name]\n"
      "{context_str}\n"
)
name_prompt = PromptTemplate(name_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
prompt_name = name_prompt.format(context_str=documents)
name = llm.complete(prompt_name)
print(name)

summary_tmpl = (
    "Extract the followings. Try to use only the "
    "information provided. "
    "Create a detailed professional summary of the applicant's resume in 10 bullet points."
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)

summary_prompt = PromptTemplate(
    summary_tmpl, prompt_type=PromptType.SUMMARY
)

prompt_summ = summary_prompt.format(context_str = documents)
summary = llm.complete(prompt_summ)
print(summary)

education_tmpl = ("Write the most recent education pursued along with year in one sentence. Strictly use only the information provided."
                  "Mention year at the beginning followed by qualification, subject (if any), name of the education institution and place of the education institution "
                  "{context_str}\n")
edu_prompt = PromptTemplate(education_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)

prompt_edu = edu_prompt.format(context_str = documents)
education = llm.complete(prompt_edu)
print(education)

techskills_tmpl = ("Write technical skills from applicant's resume. Strictly use only the information provided."
                  "Please provide a list of technical skills mentioned in the resume, including programming languages, tools, frameworks, and technologies, add side heading."
                  "List the headings in bullet points followed by skills seperated by comma along the same line."
                  "Mention headings at the beginning followed by skills seperated by comma"
                  "Make sure that heading and skills are in same line and same dict"
                  "{context_str}\n")
techskills_prompt = PromptTemplate(techskills_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
prompt_skills = techskills_prompt.format(context_str = documents)
skills = llm.complete(prompt_skills)
print(skills)

acomm_tmpl = ("Please Extract 5 significant mentioned accomplishments, achievements, awards and/or certifications in bullet points. Strictly use only the information provided."
                  "{context_str}\n")
acommplishment_prompt = PromptTemplate(acomm_tmpl, prompt_type=PromptType.SUMMARY)

prompt_acomm = acommplishment_prompt.format(context_str = documents)
accomplishments = llm.complete(prompt_acomm)
print(accomplishments)

exp_tmpl = ("Extract summary of professional experience starting from the organization name followed by extracted month , year worked for the organization and work role including duties and achievements."
            "Write it in reverse chronological order by month , year worked."
            "Provide the details in the following format:\n"
            "Organization: [Organization Name]\n"
            "Role: [Role]\n"
            "Month & Year Worked: [Month & Year]\n"
            "Description: [List of bullet points]\n"
            "{context_str}\n"
            )
exp_prompt = PromptTemplate(exp_tmpl, prompt_type=PromptType.SUMMARY)

prompt_exp = exp_prompt.format(context_str = documents)
experience = llm.complete(prompt_exp)
print(experience)











from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
def load_data(data):
    document = SimpleDirectoryReader(data).load_data()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    return document


from llama_index.llms.gemini import Gemini
def model():
    llm = Gemini(api_key= API_KEY, model = 'models/gemini-pro', temperature = 0.0)
    return llm




from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import json
llm = model()
def prompts(document):
    try:
        def extract_name():
            try:
                name_tmpl = ("Write name of the applicant in the resume."
                            "Remove any special characters between name."
                            "{context_str}\n")
                name_prompt = PromptTemplate(name_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
                prompt_name = name_prompt.format(context_str = document)
                name = llm.complete(prompt_name)
                return name.text
            except Exception as e:
                print(f"Error extracting name: {e}")
                return ""




        def extract_summary():
            try:
                summary_tmpl = ("Write summary of the professional experience and skills of the applicant in 10 comma seperated sentences without including bullet points."
                                "Focus on the most relevant and important information, excluding educational background."
                                "For example, the summary should be in the following format: \n"
                                "Senior Software Engineer with 5+ years of experience in full-stack development., Experience in creating and deploying secure APIs."
                                "\n"
                                "Please provide a concise overview of the applicant's work experience, skills, and achievements."
                                "{context_str}\n")
                summary_prompt = PromptTemplate(
                    summary_tmpl, prompt_type=PromptType.SUMMARY
                )
                prompt_summ = summary_prompt.format(context_str = document)
                summary = llm.complete(prompt_summ)
                return summary.text
            except Exception as e:
                print(f"Error extracting summary: {e}")
                return ""




        def extract_education():
            try:
                education_tmpl = ("Extract the most recently completed education in one sentence."
                                "Format the answer as: 'Year Qualification, Subject (if applicable), Institution, Location'."
                                "If the Qualification is mentioned in short form, write the full form of the qualification."
                                "If the Subject is mentioned in short form, write the full form of the subject."
                                "For example, '2017 - 2020 Bachelor of Science in Computer Science, Stanford University, California'."
                                "Strictly use only the information provided in the context."
                                "{context_str}\n")
                edu_prompt = PromptTemplate(education_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
                prompt_edu = edu_prompt.format(context_str = document)
                education = llm.complete(prompt_edu)
                return education.text
            except Exception as e:
                print(f"Error extracting education: {e}")
                return ""




        def extract_skills():
            try:
                techskills_tmpl = ("Extract skills from the resume."
                                    "Make sure the format of the extracted skills is same as that in the resume."
                                    "Provide a list of technical skills, including but not limited to programming languages, tools, frameworks, technologies, operating systems, and other relevant technical aspects."
                                    "Strictly use only the information provided in the context."
                                    "{context_str}\n")
                techskills_prompt = PromptTemplate(techskills_tmpl, prompt_type=PromptType.KEYWORD_EXTRACT)
                prompt_skills = techskills_prompt.format(context_str = document)
                skills = llm.complete(prompt_skills)
                return skills.text
            except Exception as e:
                print(f"Error extracting skills: {e}")
                return ""




        def extract_accomplishments():
            try:
                acomm_tmpl = ("Extract certifications, awards, and achievements in bullet points, such as industry-recognized certifications, prestigious awards, or notable accomplishments."
                              "If none are mentioned, simply return an empty string."
                              "Do not consider work experience as achievements."
                              "Do not include education qualifications as achievements."
                              "For example, a list of 2-5 bullet points, each describing a specific certification, award, or achievement."
                              "Strictly use only the information provided in the context."
                              "{context_str}\n")
                acommplishment_prompt = PromptTemplate(acomm_tmpl, prompt_type=PromptType.SUMMARY)
                prompt_acomm = acommplishment_prompt.format(context_str = document)
                accomplishments = llm.complete(prompt_acomm)
                return accomplishments.text
            except Exception as e:
                print(f"Error extracting accomplishments: {e}")
                return ""




        def extract_experience():
            try:
                exp_tmpl = ("Write detailed summary on professional experience in a python list."
                            "Include dictionary for each organization."
                            "'Organization', 'Role', 'Period of Work' and 'Summary' has to be the key."
                            "'Summary' key should have the values enclosed in a python list."
                            "Relevant texts has to be placed as value corresponding to its key."
                            "Make sure the output is in reverse chronological order of 'Period of Work'"
                            "{context_str}\n")
                exp_prompt = PromptTemplate(exp_tmpl, prompt_type=PromptType.SUMMARY)
                prompt_exp = exp_prompt.format(context_str = document)
                experience = llm.complete(prompt_exp)
                return experience.text
            except Exception as e:
                print(f"Error extracting experience: {e}")
                return ""




        def extract_projects():
            try:
                project_tmpl = ("Extract summary of projects from the resume and put it in a python list of dictionary."
                                "If no projects are mentioned, return an empty string."
                                "If project name or project title is mentioned then it is 'Title' else an empty string."
                                "If the role of the applicant in the project is mentioned then it is 'Role' else an empty string."
                                "Each dictionary should have the following keys: 'Title', 'Role', and 'Summary'."
                                "Populate the dictionary values with the exact text from the resume, corresponding to each key."
                                "Strictly use only the information provided in the context."
                                "{context_str}\n")
                project_prompt = PromptTemplate(project_tmpl, prompt_type=PromptType.SUMMARY)
                prompt_project = project_prompt.format(context_str = document)
                projects = llm.complete(prompt_project)
                return projects.text
            except Exception as e:
                print(f"Error extracting projects: {e}")
                return ""




        name = extract_name()
        summary = extract_summary()
        education = extract_education()
        technical_skills = extract_skills()
        accomplishments = extract_accomplishments()
        experience = extract_experience()
        projects = extract_projects()
        return name, summary, education, technical_skills, accomplishments, experience, projects
    except Exception as e:
        return " "



import re
import json
import string
def preprocessing(name, summary, education, technical_skills, accomplishments, experience, projects, position):
    try:
        def preprocess_name(text):
            try:
                # Remove special characters and escape characters
                name = re.sub(r'[^a-zA-Z\s]', '', text)
                # Convert to upper case
                name = name.upper()
                return name
            except Exception as e:
                print(f"Error preprocessing name: {e}")
                return ""




        def preprocess_summary(text):
            try:
                # Remove escape characters
                sentences = re.split(r'[\n\t]+', text)
                return sentences
            except Exception as e:
                print(f"Error preprocessing summary: {e}")
                return ""




        def preprocess_education(text):
            try:
                # Remove escape characters
                sentences = re.sub(r'[\n\t]+', '', text)
                return sentences
            except Exception as e:
                print(f"Error preprocessing education: {e}")
                return ""




        def preprocess_skills(text):
            try:
                # Remove escape characters
                sentences = re.split(r'[-\n\t]+', text)
                return sentences[1:]
            except Exception as e:
                print(f"Error preprocessing skills: {e}")
                return ""




        def preprocess_accom(text):
            try:
                # Remove escape characters
                sentences = re.split(r'[-\n\t]+', text)
                return sentences[1:]
            except Exception as e:
                print(f"Error preprocessing accomplishments: {e}")
                return sentences




        def preprocess_exp(text):
            try:
                # Remove escape characters
                sentences = re.sub(r'[\n\t]+','', text)
                return json.loads(sentences)
            except Exception as e:
                print(f"Error preprocessing experience: {e}")
                return sentences




        def preprocess_projects(text):
            try:
                if text not in '[]' or '':
                    sentences = re.sub(r'[\n\t]+', '', text)
                    return json.loads(sentences)
                else:
                    return sentences
            except Exception as e:
                print(f"Error preprocessing experience: {e}")
                return sentences




        responses = {
            'position': position,
            'name': preprocess_name(name),
            'summary': preprocess_summary(summary),
            'education': preprocess_education(education),
            'skills': preprocess_skills(technical_skills),
            'accomplishments': preprocess_accom(accomplishments),
            'experience': preprocess_exp(experience),
            'projects': preprocess_projects(projects)
        }
        return responses
    except Exception as e:
        print(f"Error: {e}")
        return ""











