# Please install OpenAI SDK first: `pip3 install openai`
import json
import os
from openai import OpenAI

ds_key = os.getenv("DEEPSEEK_API_KEY")  # To make it easier to manage

if not ds_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

math_problems_file = 'math_problems.txt'
solutions = []

try:
    with open(math_problems_file, "r") as file:
        math_problems = file.readlines()
except FileNotFoundError:
    print(f"Error: The file '{math_problems_file}' was not found.")
    exit(1)
except IOError as e:
    print(f"Error: An I/O error occurred while reading the file: {e}")
    exit(1)

for problem in math_problems:
    try:
        messages = [{"role": "user", "content": problem.strip()}]
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        solution = {
            "problem": problem.strip(),
            "reasoning": reasoning_content,
            "solution": content
        }
        
        solutions.append(solution)
        print(f"Successfully wrote the answer to the problem : {problem}")
    except Exception as e:
        print(f"Error: An error occurred while processing the problem '{problem.strip()}': {e}")
        continue

try:
    with open("math_solutions.json", "w") as output_file:
        json.dump(solutions, output_file, indent=4)

except IOError as e:
    print(f"Error: An I/O error occurred while writing to the file: {e}")
    exit(1)

######## Example ########
# To make the repo cleaner, I put the example below.
"""
math_problems.txt:
If a=0.1*e^0.1, b=1/9, c=-ln(0.9), please compare them.

math_solutions.json:
[
    {
        "problem": "If a=0.1*e^0.1, b=1/9, c=-ln(0.9), please compare them.",
        "reasoning": "Okay, so I need to compare the values of a, b, and c where:\n\na = 0.1 * e^0.1\n\nb = 1/9\n\nc = -ln(0.9)\n\nHmm, let me start by understanding each of these expressions individually. Maybe if I can approximate each one numerically, I can compare them more easily. Let's take them one by one.\n\nStarting with a = 0.1 * e^0.1. I know that e is approximately 2.71828. So e^0.1 is e raised to the power of 0.1. I remember that e^x can be approximated using the Taylor series expansion: e^x = 1 + x + x^2/2! + x^3/3! + ... for small x. Since 0.1 is a small number, maybe this will work.\n\nLet's compute e^0.1:\n\nFirst term: 1\n\nSecond term: 0.1\n\nThird term: (0.1)^2 / 2 = 0.01 / 2 = 0.005\n\nFourth term: (0.1)^3 / 6 = 0.001 / 6 \u2248 0.0001666667\n\nFifth term: (0.1)^4 / 24 \u2248 0.0001 / 24 \u2248 0.0000041667\n\nAdding these up: 1 + 0.1 = 1.1\n\n1.1 + 0.005 = 1.105\n\n1.105 + 0.0001666667 \u2248 1.1051666667\n\n1.1051666667 + 0.0000041667 \u2248 1.1051708333\n\nSo, e^0.1 \u2248 1.1051708333\n\nTherefore, a = 0.1 * 1.1051708333 \u2248 0.1105170833\n\nSo approximately 0.1105.\n\nNow, moving on to b = 1/9. Let me compute that. 1 divided by 9 is approximately 0.1111... So 0.1111 recurring. So roughly 0.1111.\n\nThen c = -ln(0.9). Let me compute ln(0.9). Since 0.9 is less than 1, the natural logarithm will be negative, so taking the negative of that will make c positive. Let's compute ln(0.9).\n\nAgain, maybe using the Taylor series expansion for ln(x) around x=1. The expansion for ln(1 - x) around x=0 is -x - x^2/2 - x^3/3 - x^4/4 - ... So here, 0.9 = 1 - 0.1, so ln(0.9) = ln(1 - 0.1). Therefore, using the expansion:\n\nln(1 - 0.1) = -0.1 - (0.1)^2/2 - (0.1)^3/3 - (0.1)^4/4 - ...\n\nSo let's compute that:\n\nFirst term: -0.1\n\nSecond term: -0.01/2 = -0.005\n\nThird term: -0.001/3 \u2248 -0.0003333333\n\nFourth term: -0.0001/4 = -0.000025\n\nFifth term: -0.00001/5 = -0.000002\n\nAdding these up:\n\n-0.1 - 0.005 = -0.105\n\n-0.105 - 0.0003333333 \u2248 -0.1053333333\n\n-0.1053333333 - 0.000025 \u2248 -0.1053583333\n\n-0.1053583333 - 0.000002 \u2248 -0.1053603333\n\nSo ln(0.9) \u2248 -0.1053603333\n\nTherefore, c = -ln(0.9) \u2248 0.1053603333\n\nSo approximately 0.10536.\n\nSo now, let me summarize the approximated values:\n\na \u2248 0.1105\n\nb \u2248 0.1111\n\nc \u2248 0.10536\n\nTherefore, comparing these numbers:\n\nc is the smallest, then a, then b. So c < a < b.\n\nBut wait, let me check if my approximations are accurate enough. Maybe I should compute them with more precision or use another method.\n\nStarting with a = 0.1 * e^0.1. Let's use a calculator-like approach. Since e^0.1 is approximately 1.105170918, which is more precise. So 0.1 * 1.105170918 \u2248 0.1105170918. So a \u2248 0.110517.\n\nb = 1/9. Let's compute 1 divided by 9. 1/9 is 0.1111111111..., so approximately 0.111111.\n\nc = -ln(0.9). Let's compute ln(0.9) using a calculator. ln(0.9) \u2248 -0.105360516 (computed using a calculator). Therefore, c = 0.105360516. So c \u2248 0.1053605.\n\nSo the more precise values are:\n\na \u2248 0.110517\n\nb \u2248 0.111111\n\nc \u2248 0.1053605\n\nTherefore, ordering from smallest to largest: c < a < b.\n\nBut let me verify if my initial approximations were correct. Let's cross-validate.\n\nAlternatively, maybe there's another way to compare them without approximating. Let's see.\n\nCompare a and b first. a = 0.1 * e^0.1 and b = 1/9 \u2248 0.1111.\n\nCompute a - b = 0.1*e^0.1 - 1/9. Let's see if this is positive or negative.\n\nWe can compute 0.1*e^0.1 \u2248 0.110517 and 1/9 \u2248 0.111111, so 0.110517 - 0.111111 \u2248 -0.000594, which is negative. Therefore, a < b.\n\nNow compare a and c. a \u2248 0.1105 and c \u2248 0.10536, so a > c.\n\nTherefore, c < a < b.\n\nAlternatively, maybe we can find exact relationships or inequalities without numerical approximation.\n\nStarting with c = -ln(0.9). Let's recall that ln(1 - x) \u2248 -x - x^2/2 - x^3/3 - ... for |x| < 1.\n\nBut 0.9 = 1 - 0.1, so ln(0.9) = ln(1 - 0.1) \u2248 -0.1 - 0.01/2 - 0.001/3 - 0.0001/4 - ... as before. So c = -ln(0.9) \u2248 0.1 + 0.005 + 0.000333... + 0.000025 + ... \u2248 0.10536, which is approximately 0.10536.\n\nCompare this with a = 0.1 * e^0.1. Let's write e^0.1 as 1 + 0.1 + 0.01/2 + 0.001/6 + 0.0001/24 + ... \u2248 1.1051708...\n\nTherefore, multiplying by 0.1 gives 0.11051708...\n\nSo a is approximately 0.1105, which is higher than c's 0.10536. So a > c.\n\nSimilarly, b = 1/9 is approximately 0.1111, which is higher than a's 0.1105. So a < b.\n\nTherefore, the order is c < a < b.\n\nAlternatively, is there a way to compare these without using approximations? Let's think.\n\nFirst, comparing a and b:\n\na = 0.1 * e^{0.1}\n\nb = 1/9 \u2248 0.1111\n\nWe can write 0.1 * e^{0.1} vs 1/9.\n\nMultiply both sides by 10: e^{0.1} vs 10/9 \u2248 1.1111\n\nWe know that e^{0.1} is approximately 1.10517, which is less than 10/9. Therefore, e^{0.1} < 10/9, so 0.1 * e^{0.1} < 1/9. Therefore, a < b.\n\nSimilarly, comparing a and c:\n\na = 0.1 * e^{0.1}\n\nc = -ln(0.9)\n\nWe can write c = -ln(0.9) = ln(1/0.9) = ln(10/9) \u2248 0.10536\n\nCompare a = 0.1 * e^{0.1} \u2248 0.1105 vs c \u2248 0.10536. So a > c.\n\nAlternatively, is there a way to relate these without approximating?\n\nLet me consider the function f(x) = e^x - 1/(1 - x) for x < 1. Maybe not directly helpful.\n\nAlternatively, consider the inequality e^x > 1 + x for x \u2260 0. So e^{0.1} > 1.1. Then 0.1 * e^{0.1} > 0.11. Similarly, c = -ln(0.9) = ln(10/9). Let's compute ln(10/9). 10/9 \u2248 1.1111, so ln(1.1111). The natural logarithm of 1.1111 is approximately 0.10536. So as per earlier.\n\nAlternatively, since we can use the inequality ln(1 + x) < x for x > -1, x \u2260 0. Here, ln(10/9) = ln(1 + 1/9) < 1/9 \u2248 0.1111. But c = ln(10/9) \u2248 0.10536 < 1/9. So c < 1/9. But a = 0.1 * e^{0.1} \u2248 0.1105 < 1/9 \u2248 0.1111? Wait, 0.1105 is less than 0.1111? Wait, 0.1105 is approximately 0.1105, and 1/9 is 0.1111..., so yes, a < b. And c \u2248 0.10536 < a. Therefore, c < a < b.\n\nAlternatively, perhaps we can use more precise inequalities.\n\nFor example, for x > 0, e^x < 1 + x + x^2. Let's check with x = 0.1.\n\n1 + 0.1 + 0.01 = 1.11. e^0.1 \u2248 1.10517, which is less than 1.11. Therefore, e^0.1 < 1 + 0.1 + 0.01 = 1.11. Therefore, 0.1 * e^0.1 < 0.1 * 1.11 = 0.111. But 1/9 is approximately 0.111111..., so 0.111 < 0.1111..., so a < b.\n\nSimilarly, for ln(10/9). Let's use the inequality that for x > 0, ln(1 + x) < x - x^2/2 + x^3/3. Wait, the Taylor series for ln(1 + x) is x - x^2/2 + x^3/3 - x^4/4 + ... for |x| < 1. Since 10/9 = 1 + 1/9, so x = 1/9. Therefore, ln(10/9) = ln(1 + 1/9) = 1/9 - (1/9)^2/2 + (1/9)^3/3 - (1/9)^4/4 + ...\n\nCompute the first few terms:\n\nFirst term: 1/9 \u2248 0.1111\n\nSecond term: - (1/81)/2 \u2248 -0.0061728395\n\nThird term: (1/729)/3 \u2248 0.0004572474\n\nFourth term: - (1/6561)/4 \u2248 -0.0000380656\n\nFifth term: (1/59049)/5 \u2248 0.0000033905\n\nAdding up:\n\n0.1111 - 0.0061728395 \u2248 0.1049271605\n\n0.1049271605 + 0.0004572474 \u2248 0.1053844079\n\n0.1053844079 - 0.0000380656 \u2248 0.1053463423\n\n0.1053463423 + 0.0000033905 \u2248 0.1053497328\n\nSo ln(10/9) \u2248 0.1053497328, which is approximately 0.10535, close to our previous calculation. Therefore, c \u2248 0.10535.\n\nCompare with a = 0.1 * e^0.1 \u2248 0.110517. So indeed, c < a.\n\nBut how to compare a and c without approximating? Let's see.\n\nWe have a = 0.1 * e^{0.1} and c = ln(10/9). Let's set them equal and see if we can find a relationship.\n\n0.1 * e^{0.1} = ln(10/9)\n\nMultiply both sides by 10:\n\ne^{0.1} = 10 * ln(10/9)\n\nBut e^{0.1} \u2248 1.10517 and 10 * ln(10/9) \u2248 10 * 0.10536 \u2248 1.0536, which is not equal, so that doesn't help.\n\nAlternatively, maybe consider the function f(x) = e^{x} - 10 * ln(10/(10 - x)) for x = 1? Wait, maybe not.\n\nAlternatively, consider the difference a - c = 0.1 e^{0.1} - ln(10/9). Let's see if this is positive.\n\nFrom approximate values, 0.1 e^{0.1} \u2248 0.1105, ln(10/9) \u2248 0.10536, so difference \u2248 0.00514, which is positive. Therefore, a > c.\n\nAlternatively, perhaps use the inequality e^x > 1 + x + x^2/2 for x > 0.\n\nFor x = 0.1, e^{0.1} > 1 + 0.1 + 0.005 = 1.105. Therefore, 0.1 * e^{0.1} > 0.1 * 1.105 = 0.1105.\n\nOn the other hand, ln(10/9) = ln(1 + 1/9). Using the inequality ln(1 + x) < x - x^2/2 + x^3/3 for x > 0.\n\nSo ln(1 + 1/9) < 1/9 - (1/9)^2/2 + (1/9)^3/3.\n\nCalculate that:\n\n1/9 \u2248 0.1111\n\n(1/9)^2 = 1/81 \u2248 0.012345679, divided by 2 is \u2248 0.0061728395\n\n(1/9)^3 = 1/729 \u2248 0.0013717421, divided by 3 is \u2248 0.0004572474\n\nSo ln(1 + 1/9) < 0.1111 - 0.0061728395 + 0.0004572474 \u2248 0.1053844079\n\nSo ln(10/9) < 0.1053844079\n\nWhereas 0.1 * e^{0.1} > 0.1105 as per above.\n\nTherefore, 0.1105 > 0.10538, so a > c.\n\nHence, combining the inequalities:\n\nc < a < b.\n\nTherefore, the order from smallest to largest is c, a, b.\n\n**Final Answer**\nThe values from smallest to largest are \\( c \\), \\( a \\), and \\( b \\). Thus, the comparison is \\boxed{c < a < b}.",
        "solution": "To compare the values of \\( a = 0.1 \\cdot e^{0.1} \\), \\( b = \\frac{1}{9} \\), and \\( c = -\\ln(0.9) \\):\n\n1. **Calculate \\( a \\):**\n   - Using the Taylor series expansion for \\( e^{0.1} \\):\n     \\[\n     e^{0.1} \\approx 1 + 0.1 + \\frac{0.01}{2} + \\frac{0.001}{6} + \\frac{0.0001}{24} \\approx 1.1051708333\n     \\]\n   - Therefore, \\( a = 0.1 \\cdot 1.1051708333 \\approx 0.1105170833 \\).\n\n2. **Calculate \\( b \\):**\n   - \\( \\frac{1}{9} \\approx 0.1111111111 \\).\n\n3. **Calculate \\( c \\):**\n   - Using the Taylor series expansion for \\( \\ln(0.9) \\):\n     \\[\n     \\ln(0.9) = \\ln(1 - 0.1) \\approx -0.1 - \\frac{0.01}{2} - \\frac{0.001}{3} - \\frac{0.0001}{4} \\approx -0.1053603333\n     \\]\n   - Therefore, \\( c = -\\ln(0.9) \\approx 0.1053603333 \\).\n\n4. **Compare the values:**\n   - Approximated values:\n     - \\( a \\approx 0.110517 \\)\n     - \\( b \\approx 0.111111 \\)\n     - \\( c \\approx 0.1053605 \\)\n   - Using inequalities and series expansions confirms the order:\n     - \\( c \\) is the smallest, followed by \\( a \\), then \\( b \\).\n\nThus, the comparison is \\(\\boxed{c < a < b}\\)."
    }
]
"""