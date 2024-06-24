from llama_index.core.evaluation import RelevancyEvaluator, generate_question_context_pairs, FaithfulnessEvaluator, \
    BatchEvalRunner
from traits.trait_types import self


class DocuEval:
    def __init__(self, index, query_engine, workers=8):
        self.evaluator_relevancy = RelevancyEvaluator()
        self.eval_Faithfulness = FaithfulnessEvaluator()
        self.index = index
        self.workers = workers
        self.query_engine = query_engine

    def run_evaluation(self, questions):
        total_score_relevancy = 0
        total_score_faithfulness = 0

        for count, query in enumerate(questions):
            print("Question:" + str(count) + " Query:" + str(query))
            print("Do the query")
            response = self.query_engine.query(query[0])
            # response = query_engine.query("How to init and get data from analog inputs on TTC 500 in C ?")
            print(response)

            print("Starting eval")

            # print("Generate question context pairs...")
            # qa_dataset = generate_question_context_pairs(nodes, llm=Settings.llm, num_questions_per_chunk=2)

            eval_result = self.evaluator_relevancy.evaluate_response(query=query, response=response)
            # print(str(eval_result))
            print("Relevancy Score:" + str(eval_result.score))
            total_score_relevancy += eval_result.score

            eval_faithfull_result = self.eval_Faithfulness.evaluate_response(query=query, response=response)
            print("Faithfulness Score:" + str(eval_faithfull_result.score))
            total_score_faithfulness += eval_faithfull_result.score
            print("Total Relevancy Score:" + str(total_score_relevancy / (count+1)))

            print("Total Faithfulness Score:" + str(total_score_faithfulness / (count+1)))
        return {"faithfulness": total_score_faithfulness / len(questions),
                "relevancy": total_score_relevancy / len(questions)}
        # runner = BatchEvalRunner({"faithfulness": self.eval_Faithfulness, "relevancy": self.evaluator_relevancy},
        #                              workers=self.workers)
        # return await runner.aevaluate_queries(
        #    self.index, queries=questions
        # )

    # def get_eval_results(self, key, eval_results):
    #     results = eval_results[key]
    #     correct = 0
    #     for result in results:
    #         if result.passing:
    #             correct += 1
    #     score = correct / len(results)
    #     print(f"{key} Score: {score}")
    #     return score
    #
    # def get_total_faithfulness_results(self, eval_results):
    #     return self.get_eval_results("faithfulness", eval_results)
    #
    # def get_total_relevancy_results(self, eval_results):
    #     return self.get_eval_results("relevancy", eval_results)
