# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


class LogprobsLists(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: list[list[int]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: list[list[float]]
    # [num_reqs]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )


@dataclass
class SamplerOutput:

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # INVALID_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Optional[LogprobsTensors]


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:

    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]]

    # num_reqs x num_spec_tokens
    spec_token_ids: Optional[list[list[int]]]

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
    req_ids=[],
    req_id_to_index={},
    sampled_token_ids=[],
    spec_token_ids=None,
    logprobs=None,
    prompt_logprobs_dict={},
)

# Supports the uncertainty estimation for TFB.
class UncertaintyLists(NamedTuple):

    # same as logprob_token_ids, might be redundant
    # [num_reqs, max_num_logprobs + 1]
    uncertainty_token_ids: list[list[int]] 

    # [num_reqs, max_num_logprobs + 1]
    total_uncertainties: list[list[float]]
    # [num_reqs, max_num_logprobs + 1]
    aleatoric_uncertainties: list[list[float]]
    # [num_reqs, max_num_logprobs + 1]
    epistemic_uncertainties: list[list[float]]

    def slice(self, start: int, end: int):
        return UncertaintyLists(
            self.uncertainty_token_ids[start:end],
            self.total_uncertainties[start:end],
            self.aleatoric_uncertainties[start:end],
            self.epistemic_uncertainties[start:end],
        )

@dataclass 
class ModelRunnerOutputWithUncertainty(ModelRunnerOutput):
    
    # [num_reqs, max_num_logprobs + 1]
    uncertainties: Optional[UncertaintyLists]


EMPTY_MODEL_RUNNER_OUTPUT_WITH_UNCERTAINTY = ModelRunnerOutputWithUncertainty(
    req_ids=[],
    req_id_to_index={},
    sampled_token_ids=[],
    spec_token_ids=None,
    logprobs=None,
    prompt_logprobs_dict={},
    # TFB uncertainty estimation
    uncertainties=None
)