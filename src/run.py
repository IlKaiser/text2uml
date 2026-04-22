#!/usr/bin/env python3
"""
Unified text-to-UML runner.

Usage:
    python run.py                      # uses config.yaml next to this file
    python run.py --config my.yaml
"""

import argparse
import gc
import logging
import os
import sys
import time
import threading
import _thread as thread
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()
log = logging.getLogger(__name__)


def _setup_logging(log_file=None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", mode="a"))
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        handlers=handlers,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

# ── Zero-Shot ─────────────────────────────────────────────────────────────────
_ZERO_SHOT_SYSTEM = """\
You will be asked by the user to create a plant UMl model from specification text. Do so in the most
clear way possible, avoid class properties and assign molteplicity.

Do include attributes for classes. For example the class Book would be:

class Book{{ String Title, String Author, Date PublicationDate }}

Use only bi-directional arc for relations and no description. For example a relation between
the class Book and the class Page, if the Book can have from one to many pages and the
pages could have exactly one book, would be:

Book "1..1" -- "1..*" Page

Adapt the cardinality to each case. Where no specific cardinality is specified, use the default "0..*".
If necessary, feel free to use inheritance.
The plantuml has to be the class diagram. In generating the diagram perform this steps in order

1. Extract class from text
2. Extract attributes for each class
3. Extract relations form text and look for the inheritance
4. Assign the relation to the corresponding class
5. Add cardinality to the relations

Put everything in this order: first all classes and then all relations. In our example would be:

@startuml

class Book{{ String Title, String Author, Date PublicationDate }}
class Page{{ String Content}}

Book "1..1" -- "1..*" Page

@enduml

Output plantuml without futher text or explaination."""

_PROMPT_ZERO_SHOT = ChatPromptTemplate([
    ("system", _ZERO_SHOT_SYSTEM),
    ("user", "Transform into plant uml this specification text: {text}"),
])

# ── Shot base (shared header for One-Shot and Few-Shot) ───────────────────────
_SHOT_BASE = """\
You will be asked by the user to create a plant UMl model from specification text. Do so in the most
clear way possible, avoid class properties and assign molteplicity.

Do include attributes for classes. For example the class Book would be:

class Book{{ String Title, String Author, Date PublicationDate }}

Use only bi-directional arc for relations and no description. For example a relation between
the class Book and the class Page, if the Book can have from one to many pages and the
pages could have exactly one book, would be:

Book "1..1" -- "1..*" Page

Adapt the cardinality to each case. Where no specific cardinality is specified, use the default "0..*".
If necessary, feel free to use inheritance.
The plantuml has to be the class diagram. In generating the diagram perform this steps in order

1. Extract class from text
2. Extract attributes for each class
3. Extract relations form text and look for the inheritance
4. Assign the relation to the corresponding class
5. Add cardinality to the relations

Put everything in this order: first all classes and then all relations. In our example would be:

@startuml

class Book{{ String Title, String Author, Date PublicationDate }}
class Page{{ String Content}}

Book "1..1" -- "1..*" Page

@enduml

Output plantuml without futher text or explaination."""

_INSURANCE_SPEC = """\
Alpha Insurance is an insurance company that provides its clients with various types of insurance policies.
As soon as a customer addresses Alpha Insurance, a broker is assigned to follow the customer's file.
The broker is registered in the system, so that when a customer calls, based on the contract,
the help desk can immediately trace who is the customer's first account manager.
After the broker is assigned to the customer, the latter indicates which type(s) of insurance policy they would like to sign for,
so the broker could, depending on the case, either assess the customer's profile on the spot or send the customer's file
for analysis to the head office. After the customer's profile has been assessed and the customer has been deemed trustworthy,
a preliminary contract/offer on an insurance product is made to the customer either in person or by email.
(Such offers can also be extended to already existing customers.)
If the customer agrees to the offer, the contract is signed by both parties.
After the signing of the contract, the client enjoys the coverage and is invoiced
(monthly or yearly - depending on the choice made in the contract) according to the price of the insurance product they bought.

In case the insured event happens, a customer should send a claim for compensation.
Then the company opens one or several claim cases (e.g. in case of an accident,
often material damage & physical damage are handled separately). Once the case file is complete,
it is sent for assessment by different estimators based on their area of expertise.
According to the reports issued by the estimators, it is decided whether the claim case is approved.
In case of approval the compensation decision is registered that stipulates which costs are eligible for (partial) refund.
For the supplied documents, the sum of compensation is calculated and the compensation is paid to the customer's account.

The estimators' reports must be stored in the database for at least one year after the payment of compensation for legal purposes."""

_INSURANCE_UML = """\
@startuml

class Customer {{
    String Name
    String email
}}
class InsurancePolicy {{
    String Type
    Double MonthlyPrice
    Double YearlyPrice
}}
class Contract {{
    String Status
    String InvoiceFrequency
}}
class Invoice {{}}
class Broker {{
    String Name
}}
class Claim {{
    Double CompensationTotal
    Double calculateCompenstationSum()
}}
class ClaimCase {{
    String Status
    String CompensationDecision
}}
class Report {{}}
class Estimator {{
    String AreaOfExpertise
}}
class CompensationPayment {{
    Date PaymentDate
}}

Customer "1"--"0..*" Contract
Contract "1" -- "0..*" Invoice
InsurancePolicy "1" -- "0..*" Contract
Contract "1" -- "0..*" ClaimCase
Customer "1" -- "0..*" Claim
Claim "1" -- "0..*" ClaimCase
ClaimCase "1" -- "0..*" CompensationPayment
Estimator "0..*" - "0..*" ClaimCase
(ClaimCase,Estimator) .. Report
Customer "0..*" - "0..1" Broker

@enduml"""

_GASSTATION_SPEC = """\
An oil company possesses a number of gas stations spread across the country. Customers can pay their refuel turn in
cash at the counter or with a personal credit card, but they can also ask for a refuel card.
When using their refuel card, customers do not pay the refuel turn immediately.
Rather, at the end of each month, the holder of the fuel card receives a monthly invoice with the amounts of
all the refuel turns paid with the fuel card. On the monthly invoice discounts are given depending on the turnover
of the customer during the invoiced month. Invoicing and paying occurs in any order:
one can already receive a second invoice before having paid the first one.
But obviously, each invoice must first have been sent before a payment can be received.
Each refuel turn only appears on the invoice of the month of the refuel turn. If a customer doesn't pay the invoice,
the unpaid refuel turns are not added to the next month's invoice. Instead, the invoice is simply sent again to the customer to
remind him/her of the payment due. If a customer repeatedly does not pay the invoices (e.g. two months in a row),
then the customer is suspended and his/her card is blocked until the bills have been paid.
The exact threshold to block a customer is not 100% fixed: very good customers are given a bit more slack than new customers.
If the bills are settled, a customer may return to the normal state.
Each gas station has a number of pumps that each contain a particular type of gasoline.
Obviously only one customer can refuel his/her car at a time at a given pump. When the nozzle is taken,
the refueling can start. When the nozzle is put back the refueling is stopped. Between those two moments,
no other customer can use this pump. As soon as the nozzle is put back, the pump becomes available for the next customer.
Each pump has a separate reservoir that must be refilled when the quantity of gasoline drops below a certain preset reordering level.
Refilling can occur at any time.
The system should produce a daily report indicating which pump needs to be refilled and for each pump the current level of
gasoline and the quantity to order. As the bottom of a tank can contain dirt, when the fuel drops below a critical level,
the pump is automatically blocked so that no refueling can occur with this pump. For the same reason,
when refilling a tank that was still above the critical level, the pump is temporarily blocked by the gas station responsible for
approximatively an hour, so that dirt can sink to the bottom. The pump is released when the gas station responsible deems that
refueling with this pump is safe again."""

_GASSTATION_UML = """\
@startuml

class CardHolder {{
    String Name
    Boolean Suspended
}}
class Invoice {{
    Int Number
    Double Discount
    String status
}}
class InvoiceLine {{
    Int Number
}}
class GasStation {{
    String Name
}}
class Pump {{
    String Type
    Boolean InUse
    Double RefillThreshold
    Boolean Blocked
}}
class RefuelTurn {{
    Int RefuelTurnNumber
}}
class CashTurn {{}}

Pump "1" -- "0..*" RefuelTurn
Pump "1" -- "0..*" CashTurn
GasStation "1" -- "0..*" Pump
CardHolder "1" -- "0..*" Invoice
CardHolder "1" -- "0..*" RefuelTurn
Invoice "1" -- "0..*" InvoiceLine
RefuelTurn "1" -- "0..1" InvoiceLine

@enduml"""

_PROMPT_ONE_SHOT = PromptTemplate.from_template(
    _SHOT_BASE + """

##############

Here is an example of how should be done.

The specificartion example text is:

""" + _INSURANCE_SPEC + """

The corresponding uml is:

""" + _INSURANCE_UML + """

##############

The specification text is:

{text}

##############

Based on the example, the uml output is:
""")

_PROMPT_FEW_SHOT = PromptTemplate.from_template(
    _SHOT_BASE + """

##############

Here is an example of how should be done.

The specificartion example text is:

""" + _INSURANCE_SPEC + """

The corresponding uml is:

""" + _INSURANCE_UML + """

################

Here is another example:

Specification text:

""" + _GASSTATION_SPEC + """

The plant uml is:

""" + _GASSTATION_UML + """

##############

The specification text is:

{text}

##############

Based on the example, the uml output is:
""")

# ── CoT prompts ───────────────────────────────────────────────────────────────
_COT_CLASS = """\
Step 1:

You will be asked to extact, from text specification, the list of class to use in
a uml conceptual model diagram. Include only classes relevant for conceptual modeling.

Choose the key words from text and output the results in a list form:

[ClassName1, ClassName2, ...]

i.e. for a Book text specification:

[Book, Page, Author, ...]

Output only the list and no other text.
Keep the list as short as possible and do not add irrelevant classes.

##############

The specification text is:

{text}.

##############

The class list is:
"""

_COT_ASSOC = """\
Step 2:

You will be asked to extract, from text specification and a list of classes, the associations and
inheritance relationships amongst the classes to use in a UML conceptual model diagram.

**Associations** are structural links between classes (not specialization). Include only associations
relevant for conceptual modeling. Use only classes from the list. All associations are symmetrical:
if (A, B) is listed there is no need to add (B, A).

**Inheritance** represents is-a / specialization: a more specific class that is a type of a more
general class. Identify all such pairs from the text.

Output the results in two labelled sections:

ASSOCIATIONS:
[(ClassName1, ClassName2), (ClassName1, ClassName3), ...]

INHERITANCE:
[(ChildClass, ParentClass), ...]

If there are no inheritance relationships output an empty list: []

Output only the two sections and no other text.
Keep both lists as short as possible.

##############

The class list is:

{classes}

#############

The specification text is:

{text}

##############
"""

_COT_ATTR = """\
Step 2b:

You will be asked to extract, from text specification and a list of classes, the attributes
for each class to use in a UML conceptual model diagram.

For each class, list the attributes explicitly mentioned in the text together with their type.
Use simple types: String, Int, Float, Boolean, Date.

Output a dictionary-like structure:

{{ClassName1: [attr1:Type1, attr2:Type2, ...], ClassName2: [attr1:Type1], ...}}

Only include classes that have attributes explicitly mentioned in the text. Omit classes
with no attributes. If no attributes are found output: {{}}

Output only the dictionary and no other text.

##############

The class list is:

{classes}

#############

The specification text is:

{text}

##############

The attributes dictionary is:
"""

_COT_CARD = """\
Step 3:

You will be asked to extact, from text specification and a list of association, the cardinality for
each association to use in a uml conceptual model diagram.

Use only association from the list. The format of the cardinality is:

min_cardinality..max_cardinlaity

For example 1..100 has 1 as minimum cardinality and 100 as maximum cadinality.
Use the asterisk (*) for an undefined number of maximum cardinality,
and 0 for an undefined number of minumum cardinality.

If you are not sure assign default cardinality: 0..*

Decide the correct cardinality for each association from text and output the results in a list form:

[(ClassName1, Cadinality1 ,ClassName2, Cardinality2), ...]

i.e. for a Book text specification:

[(Book, 1..* ,Page, 1..1),  ...]

Output only the list and no other text.

##############

The association list is:

{association}

#############

The specification text is:

{text}

##############

The cardinality list is:
"""

_COT_PLANT = """\
Step 5:

You will be asked to create a PlantUML class diagram from:
  - a class list
  - a cardinality list (associations with multiplicities)
  - an attributes dictionary
  - an inheritance list

The format of the class list is:
[ClassName1, ClassName2, ...]

The format of the cardinality list is:
[(ClassName1, Cardinality1, ClassName2, Cardinality2), ...]

The attributes dictionary format is:
{{ClassName: [attr1:Type1, attr2:Type2, ...], ...}}

The inheritance list format is:
[(ChildClass, ParentClass), ...]

For every class in the class list write:
  class ClassName1 {{
    attr1: Type1
    attr2: Type2
  }}
Omit the curly-brace block if the class has no attributes.

For every tuple in the cardinality list write:
  ClassName1 "Cardinality1" -- "Cardinality2" ClassName2.

For every tuple in the inheritance list write:
  ChildClass --|> ParentClass

Output plantuml without further text or explanation. Use tags
@startuml
at the beginning and
@enduml
at the end of the output.

##############

The class list is:

{classes}

##############

The cardinality list is:

{card}

##############

The attributes dictionary is:

{attributes}

##############

The inheritance list is:

{inheritance}

##############

The plant uml is:
"""

# ── CoT-DomainModelGeneration prompts ─────────────────────────────────────────
_DOMAIN_NOUN = """\
Step 1:

Identify all the nouns in the specification text which can potentially be the class name.
Include as much as nouns as possible and do not care about their functions for now.

###########

The specification text is:

{text}

##############

The class list is:
"""

_DOMAIN_CLASS = """\
Step 2:

You will be asked to extact, from text specification, the list of class to use in
a uml conceptual model diagram. Include only classes relevant for conceptual modeling.

Identify classes from the nouns list extracted from the problem description above.
A class is the description for a set of similar objects that have the same structure and behavior, i.e., its instances.
All objects with the same features and behavior are instances of one class.
In general, something should be a class if it could have instances.
In general, something should be an instance if it is clearly a single member of the set defined by a class.
Keep in mind that some of the nouns may be attributes or roles of the identified classes.
Choose proper names for classes according the the following rules:
1. Noun
2. Singular
3. Not too general, not too specific - at the right level of abstraction
4. Avoid software engineering terms (data, record, table, information)
5. Conventions: first letter capitalized; camel case without spaces if needed

Example class names:
Hospital, Doctor, PartTimeEmployee

Constraints:
Create classes at the right level of abstraction.
Not all nouns in the nouns list are classes, some of them may be attributes, role names, or even not needed for diagram.
Do NOT include all the nouns list as classes. Evaluate if it is needed to be a class.
ONLY generate classes that are necessary to develop the system.

Example:
Problem Description: This system helps the Java Valley police officers keep track of the cases they are assigned to do.
Officers may be assigned to investigate particular crimes,
which involves interviewing victims at their homes and entering notes in the PI system.
Identified Class List: PoliceStation, Case, PoliceOfficer, Victim, Crime, Note

Output only the list and no other text.
Keep the list as short as possible and do not add irrelevant classes.

##############

The noun list is:

{noun}

##############

The text specification is:

{text}

##############

The class list is:
"""

_DOMAIN_ASSOC = """\
Step 3:

You will be asked to extract, from text specification and a list of classes, the associations and
inheritance relationships amongst the classes to use in a UML conceptual model diagram.

**Associations** are structural links between classes (not specialization).
Use only classes from the list.

Output associations with multiplicities using the format:
mul1 Class1 -- mul2 Class2

where mul1 and mul2 are one of: [0..*, 1..1, 0..1, 1..*]

For example:
0..* Student -- 0..5 Registration
1..1 Student -- 0..1 StudentProfile

**Inheritance** represents is-a / specialization: a more specific class that is a type of a more
general class. Identify all such pairs from the text.

Output inheritance as:
[(ChildClass, ParentClass), ...]

If there are no inheritance relationships output an empty list: []

Notes:
1. Use only the given class list. Do NOT add new classes.
2. In most cases there is only 1 relationship between the same two classes.

Output the results in two labelled sections and no other text:

ASSOCIATIONS:
mul1 Class1 -- mul2 Class2
...

INHERITANCE:
[(ChildClass, ParentClass), ...]

##############

The class list is:

{classes}

#############

The specification text is:

{text}

##############
"""

_DOMAIN_ATTR = _COT_ATTR  # same prompt

_DOMAIN_PLANT = """\
Step 5:

You will be asked to create a PlantUML class diagram from:
  - a class list
  - an association list (with multiplicities)
  - an attributes dictionary
  - an inheritance list

The format of the class list is:
ClassName1, ClassName2, ...

The association list format is:
mul1 ClassName1 -- mul2 ClassName2

The attributes dictionary format is:
{{ClassName: [attr1:Type1, attr2:Type2, ...], ...}}

The inheritance list format is:
[(ChildClass, ParentClass), ...]

For every class in the class list write:
  class ClassName1 {{
    attr1: Type1
    attr2: Type2
  }}
Omit the curly-brace block if the class has no attributes.

For every entry in the association list write:
  ClassName1 "mul1" -- "mul2" ClassName2.

For every tuple in the inheritance list write:
  ChildClass --|> ParentClass

Output plantuml without further text or explanation. Use tags
@startuml
at the beginning and
@enduml
at the end of the output.

##############

The class list is:
{classes}

##############

The association list is:

{association}

##############

The attributes dictionary is:

{attributes}

##############

The inheritance list is:

{inheritance}

##############

The plant uml is:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Chain Builders
# ─────────────────────────────────────────────────────────────────────────────

def _assoc_split_helpers():
    """Return (extract_assoc, extract_inh) lambdas that parse the two-section LLM output."""
    def extract_assoc(x):
        raw = x["assoc_raw"]
        if "ASSOCIATIONS:" in raw:
            part = raw.split("ASSOCIATIONS:")[1]
            if "INHERITANCE:" in part:
                part = part.split("INHERITANCE:")[0]
            return part.strip()
        return raw.strip()

    def extract_inh(x):
        raw = x["assoc_raw"]
        if "INHERITANCE:" in raw:
            return raw.split("INHERITANCE:")[1].strip()
        return "[]"

    return extract_assoc, extract_inh


def build_zero_shot_chain(llm):
    return _PROMPT_ZERO_SHOT | llm | StrOutputParser()


def build_one_shot_chain(llm):
    return _PROMPT_ONE_SHOT | llm | StrOutputParser()


def build_few_shot_chain(llm):
    return _PROMPT_FEW_SHOT | llm | StrOutputParser()


def build_cot_chain(llm):
    p_class = PromptTemplate(input_variables=["text"],             template=_COT_CLASS)
    p_assoc = PromptTemplate(input_variables=["text", "classes"],  template=_COT_ASSOC)
    p_attr  = PromptTemplate(input_variables=["text", "classes"],  template=_COT_ATTR)
    p_card  = PromptTemplate(input_variables=["text", "association"], template=_COT_CARD)
    p_uml   = PromptTemplate(
        input_variables=["classes", "card", "attributes", "inheritance"],
        template=_COT_PLANT,
    )

    class_chain = p_class | llm | StrOutputParser()
    assoc_chain = p_assoc | llm | StrOutputParser()
    attr_chain  = p_attr  | llm | StrOutputParser()
    card_chain  = p_card  | llm | StrOutputParser()
    uml_chain   = p_uml   | llm | StrOutputParser()

    ea, ei = _assoc_split_helpers()

    return (
        RunnableLambda(lambda x: {"text": x["text"]})
        | {"text": lambda x: x["text"], "classes": class_chain}
        | {
            "text":       lambda x: x["text"],
            "classes":    lambda x: x["classes"],
            "assoc_raw":  assoc_chain,
            "attributes": attr_chain,
        }
        | {
            "text":        lambda x: x["text"],
            "classes":     lambda x: x["classes"],
            "association": RunnableLambda(ea),
            "inheritance": RunnableLambda(ei),
            "attributes":  lambda x: x["attributes"],
        }
        | {
            "text":        lambda x: x["text"],
            "classes":     lambda x: x["classes"],
            "association": lambda x: x["association"],
            "inheritance": lambda x: x["inheritance"],
            "attributes":  lambda x: x["attributes"],
            "card":        card_chain,
        }
        | {
            "classes":     lambda x: x["classes"],
            "card":        lambda x: x["card"],
            "inheritance": lambda x: x["inheritance"],
            "attributes":  lambda x: x["attributes"],
            "uml":         uml_chain,
        }
        | RunnableLambda(lambda x: x["uml"])
        | StrOutputParser()
    )


def build_cot_domain_chain(llm):
    p_noun  = PromptTemplate(input_variables=["text"],            template=_DOMAIN_NOUN)
    p_class = PromptTemplate(input_variables=["text", "noun"],    template=_DOMAIN_CLASS)
    p_assoc = PromptTemplate(input_variables=["text", "classes"], template=_DOMAIN_ASSOC)
    p_attr  = PromptTemplate(input_variables=["text", "classes"], template=_DOMAIN_ATTR)
    p_uml   = PromptTemplate(
        input_variables=["classes", "association", "attributes", "inheritance"],
        template=_DOMAIN_PLANT,
    )

    noun_chain  = p_noun  | llm | StrOutputParser()
    class_chain = p_class | llm | StrOutputParser()
    assoc_chain = p_assoc | llm | StrOutputParser()
    attr_chain  = p_attr  | llm | StrOutputParser()
    uml_chain   = p_uml   | llm | StrOutputParser()

    ea, ei = _assoc_split_helpers()

    return (
        RunnableLambda(lambda x: {"text": x["text"]})
        | {"text": lambda x: x["text"], "noun": noun_chain}
        | {"text": lambda x: x["text"], "noun": lambda x: x["noun"], "classes": class_chain}
        | {
            "text":       lambda x: x["text"],
            "classes":    lambda x: x["classes"],
            "assoc_raw":  assoc_chain,
            "attributes": attr_chain,
        }
        | {
            "classes":     lambda x: x["classes"],
            "association": RunnableLambda(ea),
            "inheritance": RunnableLambda(ei),
            "attributes":  lambda x: x["attributes"],
        }
        | {
            "classes":     lambda x: x["classes"],
            "association": lambda x: x["association"],
            "inheritance": lambda x: x["inheritance"],
            "attributes":  lambda x: x["attributes"],
            "uml":         uml_chain,
        }
        | RunnableLambda(lambda x: x["uml"])
        | StrOutputParser()
    )


_CHAIN_BUILDERS = {
    "zero_shot":  build_zero_shot_chain,
    "one_shot":   build_one_shot_chain,
    "few_shot":   build_few_shot_chain,
    "cot":        build_cot_chain,
    "cot_domain": build_cot_domain_chain,
}

# Result file prefix per technique (matches existing notebook convention)
_RESULT_PREFIX = {
    "zero_shot":  "",
    "one_shot":   "one_",
    "few_shot":   "few_",
    "cot":        "cot_",
    "cot_domain": "cot2_",
}

# ─────────────────────────────────────────────────────────────────────────────
# Pricing table  (USD per 1 000 tokens: input, output)
# Local providers (ollama, mlx, huggingface_local) are free – not listed here.
# ─────────────────────────────────────────────────────────────────────────────

_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":                       (0.0025,   0.010),
    "gpt-4o-mini":                  (0.00015,  0.0006),
    "gpt-4-turbo":                  (0.010,    0.030),
    "gpt-4":                        (0.030,    0.060),
    "gpt-3.5-turbo":                (0.0005,   0.0015),
    "o1":                           (0.015,    0.060),
    "o1-mini":                      (0.003,    0.012),
    "o3-mini":                      (0.0011,   0.0044),
    "o3":                           (0.010,    0.040),
    "o4-mini":                      (0.0011,   0.0044),
    "gpt-4.1":                      (0.002,    0.008),
    "gpt-4.1-mini":                 (0.0004,   0.0016),
    "gpt-4.1-nano":                 (0.0001,   0.0004),
    "gpt-5-mini":                   (0.00025,  0.002),
    "gpt-5-nano":                   (0.00005,  0.0004),
    "gpt-5.2":                      (0.000875, 0.007),
    # Anthropic
    # HuggingFace / Microsoft
    "phi-4":                        (0.00006,  0.00014),
    # Anthropic
    "claude-3-5-sonnet-20241022":   (0.003,    0.015),
    "claude-3-5-haiku-20241022":    (0.0008,   0.004),
    "claude-3-opus-20240229":       (0.015,    0.075),
    "claude-3-haiku-20240307":      (0.00025,  0.00125),
    "claude-3-7-sonnet-20250219":   (0.003,    0.015),
    "claude-haiku-4-5-20251001":    (0.0008,   0.004),
    "claude-sonnet-4-5-20250929":   (0.003,    0.015),
    "claude-sonnet-4-6":            (0.003,    0.015),
    "claude-opus-4-6":              (0.015,    0.075),
    # Mistral
    "mistral-large-latest":         (0.002,    0.006),
    "mistral-large-2411":           (0.002,    0.006),
    "mistral-small-latest":         (0.0002,   0.0006),
    "mistral-small-2506":           (0.0002,   0.0006),
    "codestral-2508":               (0.0003,   0.0009),
    "open-mistral-7b":              (0.00025,  0.00025),
    # Gemini
    "gemini-1.5-pro":               (0.00125,  0.005),
    "gemini-1.5-flash":             (0.000075, 0.0003),
    "gemini-2.0-flash":             (0.0001,   0.0004),
    "gemini-2.0-flash-lite":        (0.000075, 0.0003),
    "gemini-2.5-flash":             (0.00015,  0.0006),
    "gemini-2.5-pro":               (0.00125,  0.010),
    # DeepSeek
    "deepseek-chat":                (0.00014,  0.00028),
    "deepseek-reasoner":            (0.00055,  0.00219),
}


def _cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost; 0.0 if model not in pricing table."""
    key = model.split("/")[-1]  # strip org prefix if present
    price = _PRICING.get(model) or _PRICING.get(key)
    if not price:
        return 0.0
    in_rate, out_rate = price
    return (input_tokens * in_rate + output_tokens * out_rate) / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# MLX → HuggingFace model mapping
# Maps mlx-community quantised model IDs to their unquantised HuggingFace originals.
# Used when huggingface_local.from_mlx: true is set in config.yaml.
# ─────────────────────────────────────────────────────────────────────────────

MLX_TO_HF: dict = {
    # ── Meta Llama ───────────────────────────────────────────────────────────
    "mlx-community/Llama-3.2-3B-Instruct-4bit":           "meta-llama/Llama-3.2-3B-Instruct",
    "mlx-community/Llama-3.2-1B-Instruct-4bit":           "meta-llama/Llama-3.2-1B-Instruct",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit":        "meta-llama/Meta-Llama-3-8B-Instruct",
    # ── Microsoft Phi ────────────────────────────────────────────────────────
    "mlx-community/phi-4-8bit":                            "microsoft/phi-4",
    "mlx-community/Phi-4-8bit":                            "microsoft/phi-4",
    "mlx-community/Phi-3.5-mini-instruct-4bit":           "microsoft/Phi-3.5-mini-instruct",
    "mlx-community/Phi-3-mini-4k-instruct-4bit":          "microsoft/Phi-3-mini-4k-instruct",
    # ── Mistral ──────────────────────────────────────────────────────────────
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit":        "mistralai/Mistral-7B-Instruct-v0.3",
    "mlx-community/Mistral-7B-Instruct-v0.2-4bit":        "mistralai/Mistral-7B-Instruct-v0.2",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit":      "mistralai/Mistral-Nemo-Instruct-2407",
    # ── TII Falcon ───────────────────────────────────────────────────────────
    "mlx-community/Falcon3-10B-Instruct-6bit":            "tiiuae/Falcon3-10B-Instruct",
    "mlx-community/Falcon3-7B-Instruct-4bit":             "tiiuae/Falcon3-7B-Instruct",
    "mlx-community/Falcon3-3B-Instruct-4bit":             "tiiuae/Falcon3-3B-Instruct",
    # ── Google Gemma ─────────────────────────────────────────────────────────
    "mlx-community/gemma-3-4b-it-5bit":                   "google/gemma-3-4b-it",
    "mlx-community/gemma-3-1b-it-4bit":                   "google/gemma-3-1b-it",
    "mlx-community/gemma-3-12b-it-4bit":                  "google/gemma-3-12b-it",
    "mlx-community/gemma-2-2b-it-4bit":                   "google/gemma-2-2b-it",
    "mlx-community/gemma-2-9b-it-4bit":                   "google/gemma-2-9b-it",
    # ── Qwen ─────────────────────────────────────────────────────────────────
    "mlx-community/Qwen2.5-3B-Instruct-4bit":             "Qwen/Qwen2.5-3B-Instruct",
    "mlx-community/Qwen2.5-7B-Instruct-4bit":             "Qwen/Qwen2.5-7B-Instruct",
    "mlx-community/Qwen2.5-14B-Instruct-4bit":            "Qwen/Qwen2.5-14B-Instruct",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510": "Qwen/Qwen2.5-3B-Instruct",
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit":     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit":    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM Factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(provider: str, model: str, pcfg: dict):
    """Instantiate the appropriate LangChain LLM for (provider, model)."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model}
        if pcfg.get("reasoning_effort") and model.startswith("o"):
            kwargs["reasoning_effort"] = pcfg["reasoning_effort"]
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=4096, timeout=None, max_retries=2)

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=model, temperature=0, max_tokens=None, timeout=None, max_retries=2)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=pcfg.get("temperature", 0), timeout=pcfg.get("timeout", 3))

    if provider == "huggingface":
        endpoint_url = pcfg.get("endpoints", {}).get(model)
        if endpoint_url:
            # Dedicated endpoint: use OpenAI-compatible client so the model name
            # is passed correctly in the request body.
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url=endpoint_url.rstrip("/") + "/v1",
                api_key=os.environ.get("HF_TOKEN", pcfg.get("token", "")),
                temperature=0,
                max_tokens=pcfg.get("max_new_tokens", 2048),
            )
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=pcfg.get("max_new_tokens", 2048),
            do_sample=False,
            repetition_penalty=1.03,
            temperature=0.01,
        )
        return ChatHuggingFace(llm=llm, verbose=False)

    if provider == "huggingface_local":
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        llm = HuggingFacePipeline.from_model_id(
            model_id=model,
            task="text-generation",
            model_kwargs={"token": os.environ.get("HF_TOKEN")},
            pipeline_kwargs={
                "max_new_tokens": pcfg.get("max_new_tokens", 2048),
                "temperature":    pcfg.get("temperature", 0.1),
                "do_sample":      pcfg.get("do_sample", False),
            },
        )
        chat = ChatHuggingFace(llm=llm, verbose=False)
        # Auto-disable thinking for models whose chat template supports it (e.g. Qwen3).
        # Without this, thinking models spend the entire timeout budget on <think> chains
        # and never emit the actual answer.
        _tmpl = getattr(chat.tokenizer, "chat_template", "") or ""
        if "enable_thinking" in _tmpl:
            _orig_apply = chat.tokenizer.apply_chat_template
            chat.tokenizer.apply_chat_template = (
                lambda *a, **kw: _orig_apply(*a, **{**kw, "enable_thinking": False})
            )
        return chat

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model)

    if provider == "mlx":
        from langchain_community.llms.mlx_pipeline import MLXPipeline
        from langchain_community.chat_models.mlx import ChatMLX
        llm = MLXPipeline.from_model_id(
            model_id=model,
            pipeline_kwargs={"max_tokens": pcfg.get("max_tokens", 15_000), "temp": pcfg.get("temperature", 0.1)},
        )
        return ChatMLX(llm=llm)

    raise ValueError(f"Unknown provider: {provider!r}")


def _parse_hf_local_output(raw_output: str) -> str:
    """Extract the assistant's response from HuggingFace local chat-formatted output.

    Handles two ChatML separator styles:
      - Standard ChatML / Qwen:  <|im_start|>assistant\\n
      - Phi-4:                   <|im_start|>assistant<|im_sep|>

    Also strips any <think>…</think> reasoning block emitted by thinking models
    (e.g. Qwen3) so only the final answer is returned.
    """
    import re

    for sep in ("\n", "<|im_sep|>"):
        marker = f"<|im_start|>assistant{sep}"
        start_idx = raw_output.find(marker)
        if start_idx == -1:
            continue
        start_idx += len(marker)

        end_marker = "<|im_end|>"
        end_idx = raw_output.find(end_marker, start_idx)
        if end_idx == -1:
            end_idx = len(raw_output)

        content = raw_output[start_idx:end_idx]
        # Strip <think>...</think> block produced by thinking models
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        return content.strip()

    return raw_output  # fallback: return raw if no known marker found

def _free_vram(*objs) -> None:
    """Try to release model resources and clear GPU caches."""
    for obj in objs:
        if obj is None:
            continue
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        shutdown = getattr(obj, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        # Many LangChain wrappers hold a pipeline or underlying model attribute.
        for attr in ("llm", "pipeline", "model", "client"):
            sub = getattr(obj, attr, None)
            if sub is obj:
                continue
            if sub is not None:
                free = getattr(sub, "close", None)
                if callable(free):
                    try:
                        free()
                    except Exception:
                        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class _UsageCallback(BaseCallbackHandler):
    """Accumulates token counts across all LLM calls in a chain invocation."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def _add(self, response: LLMResult) -> None:
        out = response.llm_output or {}
        usage = (
            out.get("token_usage")      # OpenAI
            or out.get("usage")         # Anthropic / Mistral
            or {}
        )
        self.input_tokens  += usage.get("input_tokens",  0) or usage.get("prompt_tokens",     0)
        self.output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Gemini: token counts live in generation_info per generation
        for gens in response.generations:
            for gen in gens:
                info = getattr(gen, "generation_info", None) or {}
                meta = info.get("usage_metadata") or {}
                self.input_tokens  += meta.get("prompt_token_count",     0)
                self.output_tokens += meta.get("candidates_token_count", 0)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self._add(response)

    def on_chat_model_end(self, response: LLMResult, **kwargs) -> None:
        self._add(response)


def _exit_after(timeout_s: int):
    """Decorator: raise KeyboardInterrupt in main thread after timeout_s seconds."""
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(timeout_s, lambda: thread.interrupt_main())
            timer.start()
            try:
                return fn(*args, **kwargs)
            finally:
                timer.cancel()
        return inner
    return outer


def _run_chain(chain, text: str, timeout: int, callback: "_UsageCallback") -> str:
    @_exit_after(timeout)
    def _invoke():
        return chain.invoke({"text": text}, config={"callbacks": [callback]})
    try:
        return _invoke()
    except (KeyboardInterrupt, Exception):
        return ""


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def process_dataset(
    root: Path,
    chain,
    result_prefix: str,
    model: str,
    timeout: int,
    prov: str,
    force: bool = False,
    skip_blank: bool = False,
    skip_folders: set[str] | None = None,
) -> tuple[float, int, int]:
    """Run chain on every subfolder's description.md, write result_{prefix}{model}.txt.

    skip_folders: subfolder names (not paths) to exclude – used to prevent
    evaluating cases that appear verbatim as few-shot examples in the prompt.
    skip_blank:   treat existing zero-byte result files as done (skip them).

    Returns (elapsed_seconds, total_input_tokens, total_output_tokens).
    """
    safe_model = _safe_model_name(model)
    subfolders = sorted(p for p in root.iterdir() if p.is_dir())
    total_input = total_output = 0
    t0 = time.time()
    for subfolder in tqdm(subfolders, desc=f"{result_prefix}{safe_model}", leave=False):
        if skip_folders and subfolder.name in skip_folders:
            log.info("    skip example case: %s", subfolder.name)
            continue
        desc = subfolder / "description.md"
        if not desc.is_file():
            continue
        out_file = subfolder / f"result_{result_prefix}{safe_model}.txt"
        existing = out_file.exists()
        if existing:
            existing_size = out_file.stat().st_size
            if existing_size > 0 and not force:
                # Check if we need to apply parsing to existing file
                if prov == "huggingface_local":
                    existing_content = out_file.read_text(encoding="utf-8")
                    parsed_content = _parse_hf_local_output(existing_content)
                    if parsed_content != existing_content:
                        log.info("    updating existing output with parsed content: %s", out_file.relative_to(root))
                        out_file.write_text(parsed_content, encoding="utf-8")
                    else:
                        log.info("    skip existing output: %s", out_file.relative_to(root))
                else:
                    log.info("    skip existing output: %s", out_file.relative_to(root))
                continue
            if existing_size == 0:
                if skip_blank and not force:
                    log.info("    skip blank output: %s", out_file.relative_to(root))
                    continue
                if not force:
                    log.info("    existing output is empty, rerunning: %s", out_file.relative_to(root))
        text = desc.read_text(encoding="utf-8")
        cb = _UsageCallback()
        with tracing_v2_enabled():
            result = _run_chain(chain, text, timeout, cb)
        if prov == "huggingface_local":
            result = _parse_hf_local_output(result)
        total_input += cb.input_tokens
        total_output += cb.output_tokens
        out_file.write_text(result, encoding="utf-8")
    return time.time() - t0, total_input, total_output

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


from huggingface_hub import get_token, login

def main():
    parser = argparse.ArgumentParser(description="Unified text-to-UML runner")
    parser.add_argument("--config", default=None, help="Path to YAML config (default: config.yaml next to run.py)")
    parser.add_argument("--log-file", default=None, help="Path to file where logs are written")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing result files")
    parser.add_argument("--skip-blank", action="store_true", help="Skip existing zero-byte result files instead of retrying them")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    log_file = Path(args.log_file).resolve() if args.log_file else script_dir / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_file)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    script_dir = Path(__file__).parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    cfg = _load_config(config_path)

    dataset_root = (script_dir / cfg["dataset_path"]).resolve()
    timeout      = cfg.get("timeout", 360)
    techniques   = cfg.get("techniques", {})
    providers    = cfg.get("providers", {})

    if not dataset_root.is_dir():
        log.error("Dataset path not found: %s", dataset_root)
        sys.exit(1)

    # Resolve huggingface_local.from_mlx: auto-populate HF models from the mlx list
    hfl = providers.get("huggingface_local", {})
    if hfl.get("from_mlx"):
        mlx_models = providers.get("mlx", {}).get("models", [])
        derived = []
        for mlx_id in mlx_models:
            hf_id = MLX_TO_HF.get(mlx_id)
            if hf_id:
                derived.append(hf_id)
            else:
                log.warning("No HuggingFace mapping found for MLX model %r – skipping", mlx_id)
        existing = hfl.get("models") or []
        hfl["models"] = list(dict.fromkeys(existing + derived))  # merge, deduplicate, preserve order
        log.info("huggingface_local: resolved %d model(s) from mlx list", len(derived))

    # Collect (technique, provider, model) triples to run
    jobs = []
    for tech, tcfg in techniques.items():
        if not tcfg.get("enabled", False):
            continue
        if tech not in _CHAIN_BUILDERS:
            log.warning("Unknown technique %r – skipping", tech)
            continue
        for prov, pcfg in providers.items():
            if not pcfg.get("enabled", False):
                continue
            for model in pcfg.get("models", []):
                jobs.append((tech, prov, model, pcfg))

    if not jobs:
        log.warning("No enabled technique/provider/model combinations found.")
        return

    log.info("Running %d job(s) on dataset: %s", len(jobs), dataset_root)

    summary: list[tuple[str, str, str, float, int, int, float]] = []  # tech, prov, model, secs, in, out, cost

    for tech, prov, model, pcfg in tqdm(jobs, desc="jobs"):
        log.info("  %s  |  %s  |  %s", tech, prov, model)
        llm = chain = None
        try:
            llm   = _make_llm(prov, model, pcfg)
            chain = _CHAIN_BUILDERS[tech](llm)
            prefix = _RESULT_PREFIX[tech]
            tcfg_skip = set(techniques[tech].get("skip_folders", []))
            elapsed, in_tok, out_tok = process_dataset(
                dataset_root, chain, prefix, model, timeout, prov,
                force=args.force, skip_blank=args.skip_blank,
                skip_folders=tcfg_skip or None,
            )
            cost = _cost_for(model, in_tok, out_tok)
            log.info(
                "    time=%.1fs  in_tokens=%d  out_tokens=%d  cost=$%.4f",
                elapsed, in_tok, out_tok, cost,
            )
            summary.append((tech, prov, model, elapsed, in_tok, out_tok, cost))
        except Exception as exc:
            log.error("    FAILED – %s: %s", type(exc).__name__, exc)
        finally:
            _free_vram(chain, llm)
            chain = None
            llm = None

    if summary:
        log.info("─" * 80)
        log.info("%-12s %-18s %-40s %7s %8s %8s %8s",
                 "technique", "provider", "model", "secs", "in_tok", "out_tok", "cost$")
        log.info("─" * 80)
        for tech, prov, model, secs, in_tok, out_tok, cost in summary:
            log.info("%-12s %-18s %-40s %7.1f %8d %8d %8.4f",
                     tech, prov, model, secs, in_tok, out_tok, cost)
        total_cost = sum(r[6] for r in summary)

        # Per-model aggregation
        from collections import defaultdict
        model_totals: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0.0])  # [secs, in, out, cost]
        for tech, prov, model, secs, in_tok, out_tok, cost in summary:
            t = model_totals[(prov, model)]
            t[0] += secs; t[1] += in_tok; t[2] += out_tok; t[3] += cost
        log.info("─" * 80)
        log.info("Cost per model (all techniques combined):")
        log.info("%-18s %-40s %7s %8s %8s %8s",
                 "provider", "model", "secs", "in_tok", "out_tok", "cost$")
        log.info("─" * 80)
        for (prov, model), (secs, in_tok, out_tok, cost) in sorted(model_totals.items()):
            log.info("%-18s %-40s %7.1f %8d %8d %8.4f", prov, model, secs, in_tok, out_tok, cost)

        # Per-provider aggregation
        prov_totals: dict[str, list] = defaultdict(lambda: [0.0])
        for tech, prov, model, secs, in_tok, out_tok, cost in summary:
            prov_totals[prov][0] += cost
        log.info("─" * 80)
        log.info("Cost per provider:")
        for prov, (cost,) in sorted(prov_totals.items()):
            log.info("  %-18s $%.4f", prov, cost)
        log.info("─" * 80)
        log.info("Total cost: $%.4f", total_cost)

    log.info("Done.")


if __name__ == "__main__":
    main()
