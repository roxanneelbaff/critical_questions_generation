from agents.ScorerAgent import ScorerAgent


DEFAULT_PARAMETERS = {}

SUBROLE_1 = """
Specifically assess whether the question delves into the argument's core, challenging the speaker's reasoning or assumptions.

Do this in two steps:
1. First, in the argument, identify the speaker's core points, claims, and assumptions.
2. Second, assess whether the question challenges any of those.
3. Finally, reason about the usefulness of the question. The better the question adressess core points, claims and/or assumptions made in the argument, the more useful it can be considered.
"""
WEIGHT_1 = 1.0

SUBROLE_2 = """Specifically assess whether the critical question does not contain any bad reasoning, namely, questions critical positions or claims **that the speaker does not hold**. If the question contains such bad reasoning, it would render the question invalid.

Do this in three steps:
1. First, identify the speakers overall position on the topic and the claims they make.
2. Second, determine the position and claims that the question suggests the speaker holds.
3. Finally, evaluate whether there is any significant mismatch between the speaker's actual position and claims and those implied by the question. If such a mismatch exists, assess whether its significance remains acceptable, considering that this is a critical question for that argument. Reason about the usefulness of the question. Significant amounts of bad reasoning lead to a lower usefulness of the question.

Keep your answer concise overall. Don't be too strict in your evaluation.
"""
WEIGHT_2 = 1.0

SUBROLE_3 = """
Specifically assess whether the critical question is specific to the argument or rather generic. Reason about the usefulness of the question. The more concrete the question is towards the argument, the higher the usefulness. If the question could be posed for any argument, it is not useful.
"""
WEIGHT_3 = 1.0


class QuestionScorer:
    def __init__(self, llm:str):
        scorer_agent_1 = ScorerAgent(SUBROLE_1, llm, WEIGHT_1, DEFAULT_PARAMETERS)
        scorer_agent_2 = ScorerAgent(SUBROLE_2, llm, WEIGHT_2, DEFAULT_PARAMETERS)
        scorer_agent_3 = ScorerAgent(SUBROLE_3, llm, WEIGHT_3, DEFAULT_PARAMETERS)

        self.scorers = [scorer_agent_1, scorer_agent_2, scorer_agent_3]

    def score_question(self, argument: str, question: str) -> int:
        """Scores a single question from 0 to 10. Returns -1 if question can not be scored.

        Args:
            argument (str): Argument
            question (str): Critical Question

        Returns:
            int: Score from -1 to 10
        """
        scores = []
        for scorer in self.scorers:
            score = scorer.score_question(argument, question)
            scores.append((score, scorer.weight))
        weighted_average = sum(score * weight for score, weight in scores) / sum(weight for _, weight in scores)
        final_score: int = round(weighted_average)
        return final_score

    def rank_questions(self, argument: str, questions: list[str]) -> list[str]:
        """Sort questions by their usefulness score.

        Args:
            argument (str): Argument of the question
            questions (list[str]): Critical questions to sort

        Returns:
            list[str]: Sorted critical questions
        """
        scored_questions = []

        # Step 1: Score each question
        for question in questions:
            final_score = self.score_question(argument, question)
            scored_questions.append({"question": question, "score": final_score})

        # Step 2: Sort all questions by their score
        scored_questions_sorted = sorted(
            scored_questions, key=lambda x: x["score"], reverse=True
        )

        # Step 3: Return sorted questions
        return scored_questions_sorted
