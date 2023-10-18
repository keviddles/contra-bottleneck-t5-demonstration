import {
  pipeline,
  AutoTokenizer,
  AutoModelForCausalLM,
  T5PreTrainedModel,
  PreTrainedModel,
  T5Model,
  AutoConfig,
} from "@xenova/transformers";

const main = async (text: string) => {
  const modelPath = "thesephist/contra-bottleneck-t5-small-wikipedia";
  const tokenizer = await AutoTokenizer.from_pretrained(modelPath);
  const model = await AutoModelForCausalLM.from_pretrained(
    "thesephist/contra-bottleneck-t5-small-wikipedia"
  );
  // const model = await T5PreTrainedModel.from_pretrained(
  //   "thesephist/contra-bottleneck-t5-small-wikipedia"
  // );

  const embed = (text: string): Promise<unknown> => {
    console.log("embed", text);

    const inputs = tokenizer(text);
    const decoder_inputs = tokenizer("");

    console.log(model);
    return model(inputs, decoder_inputs["input_its"], true);
  };

  const generateFromLatent = async (latent: unknown) => {
    const dummy_text = ".";
    const dummy = await embed(dummy_text);
    console.log(dummy);
    // const perturb_vector = latent - dummy
  };

  const embedding = await embed(text);
  console.log(embedding);
  const reconstruction = await generateFromLatent(embedding);
  console.log(reconstruction);
};

main("Quick brown fox");
