from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    information = """
P. Douglas Bailey, Esq.

Mr. Bailey served in private venture and growth equity for life sciences and technology. A leader in civil rights protections and access for all individuals with disabilities, Mr. Bailey has extensive experience in complex and multi-district litigation and class actions involving sustainable real estate development. He studied law at U Houston Law and history at UT Austin. 

​

Joseph Kozikowski, MD

Joe served in institutional, corporate, private, and impact venture, and he led clinical, regulatory, and corporate development. A tech founder, he delivered software solutions to hundreds of life sciences companies and medical institutions. He researched medtech at MGH-MIT, and he studied corporate governance at Stanford and UCLA, strategy at JWMI, finance at Harvard, regulatory at FDLI, medicine at Johns Hopkins, and engineering at Villanova.

​

Nelia Padilla, MBA

Nelia served in institutional venture, and she led major clinical and strategic initiatives. A digital health thought leader, she delivered data-driven technology solutions to leading healthcare and life sciences companies worldwide. She studied business at Yale and biology at Harvard.

​

Stuart Pollard, PhD

Stuart served in institutional and corporate venture, and he led innovative venture models of corporate co-creation and megadeals. He led strategy, corporate development, and portfolio management at leading biotechs and growth companies. He researched molecular biology at Harvard, and he studied molecular virology and biochemistry at U Reading.

​

Anne Shehab, PhD MBA

Anne served in institutional, corporate, private, and impact venture, and she led M&A and corporate development at leading multinationals and growth companies. A medtech founder, she commercialized innovative products to medical institutions worldwide. She researched cardiovascular at Texas Heart Institute, and she studied finance at U Chicago, engineering at Rice, and science at U Edinborough.

​

Michael Warren, CQMSA

Mike led quality, regulatory, and manufacturing at leading multinationals and growth companies. An expert at the intersection of quality, regulatory, and manufacturing, he led the design, implementation, and scale-up of quality management systems for numerous, complex, first-in-class medtech and implant innovations. He studied chemistry and engineering at Boston College.
    """

    summary_template = f"""
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    # llm = ChatAnthropic(model="claude-haiku-5-5", temperature=0.25, max_tokens=200)
    llm = ChatOllama(temperature=0.25, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)
