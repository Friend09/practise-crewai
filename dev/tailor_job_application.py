from crewai import Agent, Task, Crew
import warnings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
import os
from dotenv import load_dotenv, find_dotenv
from IPython.display import Markdown, display

# This is my professional career profile (just keep updating this)
PROFESSIONAL_PROFILE_MD="/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_Interviews/subproj_Resume/files_proj_Resume/Raghu/Resume/raghuvamsi_ayapilla_resume_details_for_ai_2024.md"

def tailor_job_application(
    JOB_PROFILE: str,
    job_posting_url: str,
    github_url: str,
    personal_writeup: str,
    output_resume_file: str,
    output_interview_file: str
):
    """
    Tailor a job application by analyzing a job posting, compiling a personal profile,
    refining a resume, and preparing for interviews.

    Parameters:
    - JOB_PROFILE (str): The job profile title for the application (e.g., 'AI ML Engineer').
    - job_posting_url (str): URL of the job posting to analyze.
    - github_url (str): URL of the GitHub profile to use for extracting personal project information.
    - personal_writeup (str): A write-up providing a personal summary or introduction.
    - output_resume_file (str): The file path where the tailored resume will be saved.
    - output_interview_file (str): The file path where the interview preparation materials will be saved.

    Returns:
    - None: The function saves the tailored resume and interview preparation materials to the specified file paths.
    """
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    CLAUDEAI_API_KEY = os.getenv("CLAUDEAI_API_KEY")

    # Initialize LLM
    # llm = ChatOpenAI(model="GPT-3.5-turbo")
    # llm = ChatOpenAI(model="gpt-4-turbo")
    llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatAnthropic(model="claude-3-opus-20240229")

    # Define tools
    tool_search = SerperDevTool()
    tool_scrape = ScrapeWebsiteTool()
    tool_read_resume = FileReadTool(file_path=PROFESSIONAL_PROFILE_MD)
    tool_semantic_search_resume = MDXSearchTool(mdx=PROFESSIONAL_PROFILE_MD)

    # Define agents
    agent_researcher = Agent(
        role="Tech Job Researcher", # dont change this. this is the agent ROLE
        goal="Make sure to do amazing analysis on job posting to help job applicants",
        backstory="As a Job Researcher, your prowess in navigating and extracting critical information from job postings is unmatched. Your skills help pinpoint the necessary qualifications and skills sought by employers, forming the foundation for effective application tailoring.",
        tools=[tool_scrape, tool_search],
        llm=llm,
        verbose=True,
    )

    agent_profiler = Agent(
        role=f"Personal Profiler for {JOB_PROFILE}s",
        goal="Do incredible research on job applicants to help them stand out in the job market",
        backstory="Equipped with analytical prowess, you dissect and synthesize information from diverse sources to craft comprehensive personal and professional profiles, laying the groundwork for personalized resume enhancements.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    agent_resume_strategist = Agent(
        role=f"Resume Strategist for {JOB_PROFILE}s",
        goal=f"Find all best ways to make an {JOB_PROFILE}'s resume more professional and stand out in the current job market.",
        backstory="With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most relevant skills and experiences, ensuring they resonate perfectly with the job's requirements.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    agent_interview_preparer = Agent(
        role=f"{JOB_PROFILE}'s interview preparer",
        goal=f"Create interview questions and talking points based on the resume and job requirements.",
        backstory="Your role is crucial in anticipating the dynamics of interviews. With your ability to formulate key questions and talking points, you prepare candidates for success, ensuring they can confidently address all aspects of the job they are applying for.",
        tools=[tool_scrape, tool_search, tool_read_resume, tool_semantic_search_resume],
        llm=llm,
        verbose=True,
    )

    # Define tasks
    task_research = Task(
        description=f"Analyze the job posting URL provided ({job_posting_url}) to extract key skills, experiences, and qualifications required. Use the tools to gather content and identify and categorize the requirements.",
        expected_output="A structured list of job requirements, including necessary skills, qualifications, and experiences.",
        agent=agent_researcher,
        async_execution=True,
    )

    task_profile = Task(
        description=f"Compile a detailed personal and professional profile using the GitHub ({github_url}) URLs, and personal write-up ({personal_writeup}). Utilize tools to extract and synthesize information from these sources.",
        expected_output="A comprehensive profile document that includes skills, project experiences, contributions, interests, and communication style.",
        agent=agent_profiler,
        async_execution=True,
    )

    task_resume_strategy = Task(
        description=f"Using the profile and job requirements obtained from previous tasks, tailor the resume to highlight the most relevant areas. Employ tools to adjust and enhance the resume content. Make sure this is the best resume even BUT DO NOT make up any information. Update every section, including the initial summary, work experience, skills, and education. All to better reflect the candidate's abilities and how it matches the job posting.",
        expected_output="An updated resume that effectively highlights the candidate's abilities and experiences relevant to the job.",
        output_file=output_resume_file,
        context=[task_research, task_profile],
        agent=agent_resume_strategist,
    )

    task_interview_preparation = Task(
        description=f"Create a set of potential interview questions and talking points based on the tailored resume and job requirements. Utilize tools to generate relevant questions and discussion points. Make sure to use these questions and talking points to help the candidate highlight the main points of the resume and how it matches the job posting.",
        expected_output="A document containing key questions and talking points that the candidate should prepare for the initial interview.",
        output_file=output_interview_file,
        context=[task_research, task_profile, task_resume_strategy],
        agent=agent_interview_preparer,
    )

    # Define crew
    crew_job_application = Crew(
        agents=[
            agent_researcher,
            agent_profiler,
            agent_resume_strategist,
            agent_interview_preparer,
        ],
        tasks=[
            task_research,
            task_profile,
            task_resume_strategy,
            task_interview_preparation,
        ],
        manager_llm=llm,
        verbose=True,
    )

    # Inputs for the crew
    inputs_job_application = {
        "JOB_PROFILE": JOB_PROFILE,
        "job_posting_url": job_posting_url,
        "github_url": github_url,
        "personal_writeup": personal_writeup,
    }

    # Kickoff the crew
    result = crew_job_application.kickoff(inputs=inputs_job_application)

    # confirmation of end of script
    print("Script Ended! please check path at: `/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai/files` for generated files")

# Example usage
# Define the job profile first
JOB_PROFILE = "" # November 14, 2024 -> AI & ML Engineer

tailor_job_application(
    JOB_PROFILE=JOB_PROFILE,
    job_posting_url="https://careers.ey.com/ey/job/New-York-AI-&-Machine-Learning-Engineer-Manager-Consulting-Location-OPEN-NY-10001-8604/1120933901/?feedId=337401&utm_source=LinkedInJobPostings&utm_campaign=j2w_linkedin",
    github_url="https://github.com/Friend09",
    personal_writeup="""
As a seasoned AI/ML Engineer with over 10 years of experience in Agile Project Management, 4+ years in data science, and 10 years in business systems analysis, I am excited to bring my comprehensive expertise in advanced data and machine learning solutions to the AI/Machine Learning Engineer - Manager role at EY. My career has been defined by a commitment to leveraging cutting-edge technology and advanced analytics to solve complex challenges, drive innovation, and enhance decision-making across business functions.

Throughout my tenure at Marsh McLennan, I successfully transformed business needs into actionable, data-driven insights and machine learning models, driving significant value across various business lines. I have a deep proficiency in Python for data analysis, visualization, and API integration, which has been instrumental in developing scalable solutions, particularly for natural language processing and generative AI tasks. My expertise in frameworks like LangChain and OpenAI, as well as my experience with large language models (LLMs), positions me well to guide clients through the nuances of Generative AI and Retrieval-Augmented Generation (RAG) systems.

A project that showcases my skills and hands-on problem-solving approach is the end-to-end tag prediction model I developed for ServiceNow incidents. Using ServiceNow’s REST API, I extracted, preprocessed, and implemented an accurate tag prediction model that enhanced planning and reporting capabilities, boosting both automation and decision accuracy. This project reflects my experience designing and deploying complete ML workflows, including MLOps practices, model pipelines, and continuous monitoring—all critical for effective machine learning engineering.

In addition to technical expertise, I have extensive experience with cloud computing platforms such as AWS, which has equipped me to lead projects involving containerization, model scaling, and deploying robust ML systems. I am proficient with DevOps tools including Git, Azure DevOps, and Agile project management platforms like Jira, which enables me to guide teams in delivering complex, feature-rich AI applications efficiently. My experience with CI/CD pipelines and test-driven development further ensures that the models I implement are not only powerful but also maintainable and resilient.

My background also includes developing intelligent applications for IT operations, such as chatbots for mean time to resolution analysis and automated ticket tagging. These initiatives demonstrate my ability to deliver impactful AI-driven solutions that improve both client outcomes and internal operational efficiencies.

I am passionate about fostering a collaborative environment and thrive in client-facing roles where I can understand unique business requirements and communicate data-driven recommendations effectively. With a solid foundation in agile methodologies, I am skilled at translating complex technical concepts for diverse audiences, which is essential for the fast-paced, client-centered consulting environment at EY.

As an AI/Machine Learning Engineer at EY, I am eager to contribute my technical knowledge, strategic insight, and mentoring experience to lead teams in delivering innovative, scalable solutions that align with EY’s vision of building a better working world. My combination of technical depth, practical experience, and commitment to continuous learning will enable me to make a meaningful impact for EY’s clients and broader organizational goals.
""",
    output_resume_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai-resume/files/resume_{JOB_PROFILE}.md",
    output_interview_file=f"/Users/vamsi_mbmax/Library/CloudStorage/OneDrive-Personal/01_vam_PROJECTS/LEARNING/proj_AI/dev_proj_AI/pract-crewai-resume/files/interview_{JOB_PROFILE}.md"
)
