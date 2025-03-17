# SPDX-License-Identifier: Apache-2.0

# For Uncertainty Estimation, for TFB.

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from vllm.logger import init_logger
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from vllm.v1.engine import (
    EngineCoreOutputWithUncertainty as EngineCoreOutput, 
    EngineCoreRequest,
)
from vllm.v1.outputs import UncertaintyLists
from vllm.outputs import SampleUncertainties, Uncertainty

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class UncertaintiesProcessor:

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]

    # Logprobs for this request
    # logprobs: Optional[SampleLogprobs]
    uncertainties: Optional[SampleUncertainties]
    # prompt_logprobs: Optional[PromptLogprobs]
    # cumulative_logprob: Optional[float]
    # num_logprobs: Optional[int]
    # num_prompt_logprobs: Optional[int]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "UncertaintiesProcessor":
        return cls(
            tokenizer=tokenizer,
            uncertainties=[],
        )

    def _update_sample_uncertainties(self, uncertainties_lists: UncertaintyLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.uncertainties is not None
        token_ids_lst, total_uncertainties_lst, aleatoric_uncertainties_lst, epistemic_uncertainties_lst = uncertainties_lists

        for tus, aus, eus, token_ids in zip(total_uncertainties_lst, aleatoric_uncertainties_lst, 
                                            epistemic_uncertainties_lst, token_ids_lst):

            # Detokenize (non-incrementally).
            decoded_tokens = NONES if self.tokenizer is None else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids))

            # Update with the Logprob dictionary for this pos.
            self.uncertainties.append(
                self._make_uncertainty_dict(
                    total_uncertainties=tus,
                    aleatoric_uncertainties=aus, 
                    epistemic_uncertainties=eus, 
                    token_ids=token_ids,
                    decoded_tokens=decoded_tokens,
                ))

    @staticmethod
    def _make_uncertainty_dict(
        total_uncertainties: list[float],
        aleatoric_uncertainties: list[float],
        epistemic_uncertainties: list[float],
        token_ids: list[int],
        decoded_tokens: Iterable[Optional[str]],
    ) -> dict[int, Uncertainty]:
        """Make a Uncertainty dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
          dict[token id, Logprob]
        """

        return {
            token_id: Uncertainty(
                total_uncertainty=tu,
                aleatoric_uncertainty=au,
                epistemic_uncertainty=eu,
                decoded_token=token,
            )
            for token_id, tu, au, eu, token in zip(
                token_ids, total_uncertainties, 
                aleatoric_uncertainties, epistemic_uncertainties, 
                decoded_tokens,
            )
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_uncertainties is not None:
            self._update_sample_uncertainties(output.new_uncertainties)
