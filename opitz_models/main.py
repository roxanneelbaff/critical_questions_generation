from GeneratorAgent import GeneratorAgent

intervention = """
CLINTON: "The central question in this election is really what kind of country we want to be and what kind of future we 'll build together
Today is my granddaughter 's second birthday
I think about this a lot
we have to build an economy that works for everyone , not just those at the top
we need new jobs , good jobs , with rising incomes
I want us to invest in you
I want us to invest in your future
jobs in infrastructure , in advanced manufacturing , innovation and technology , clean , renewable energy , and small business
most of the new jobs will come from small business
We also have to make the economy fairer
That starts with raising the national minimum wage and also guarantee , finally , equal pay for women 's work
I also want to see more companies do profit-sharing"
"""

### Generator Agent ###
model_name = "llama3.2:3b-instruct-q5_K_M"  # llama3.2:1b, llama3:latest
generator = GeneratorAgent(intervention, model_name)

### Validator Agents ###
pass

### Generating critical question ###
critical_question = generator.generate_critical_question()
print(f"Critical Question:\n{critical_question}")

print("Done.")