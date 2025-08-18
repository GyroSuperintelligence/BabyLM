# baby/constants/harmony_tokens.py

# Harmony control token constants - single source of truth
# Based on o200k_harmony encoding specification

# Core Harmony tokens
START = 200006      # <|start|>
CHANNEL = 200005    # <|channel|>
MESSAGE = 200008    # <|message|>
END = 200007        # <|end|>
RETURN = 200002     # <|return|>
CALL = 200012       # <|call|>

# Reserved tokens (not used in current implementation)
CONSTRAIN = 200003  # <|constrain|>
RESERVED_200000 = 200000  # <|reserved_200000|>
RESERVED_200001 = 200001  # <|reserved_200001|>

# Token sets for different use cases

# Tokens excluded during normal generation (engine should never emit these)
GENERATION_EXCLUDED = {
    START,      # <|start|> - only for message headers
    CHANNEL,    # <|channel|> - only for message headers  
    MESSAGE,    # <|message|> - only for message headers
    CALL        # <|call|> - only state machine can emit
}

# Tokens that only the state machine in inference can emit
STATE_MACHINE_ONLY = {
    END,        # <|end|> - marks message completion
    RETURN      # <|return|> - marks inference completion
}

# All control tokens (for exclusion from orbit sweeps)
ALL_CONTROL_TOKENS = {
    RESERVED_200000,
    RESERVED_200001, 
    RETURN,
    CONSTRAIN,
    CHANNEL,
    START,
    END,
    MESSAGE,
    CALL
}