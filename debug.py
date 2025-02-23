from src.open_r1.grpo import main, TrlParser, GRPOScriptArguments, GRPOConfig, ModelConfig

if __name__ == '__main__':
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)