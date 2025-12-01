import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
# from deepseek_vl.utils.io import load_pil_images

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
# from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox

from llm.prompts import trading_prompt


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images



class TextMemory:
    def __init__(self, max_len=10):
        self.max_len = max_len
        self.memories = []

    def add(self, text: str):
        self.memories.append(text)
        if len(self.memories) > self.max_len:
            self.memories.pop(0)

    def get_all(self):
        return self.memories



class initial_llm():

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.memory = TextMemory(max_len=10)
        self.system_prompt = trading_prompt

        # ---- FIX 1: load model only once ----
        self.dtype = torch.bfloat16

        # specify the path to the model
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        self.vl_gpt = self.vl_gpt.cuda().eval()

        # self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        # self.tokenizer = self.vl_chat_processor.tokenizer
        # self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        #     model_path, trust_remote_code=True
        # )
        # self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()


    def build_conversation(self, user_query: str, image_paths=None):

        # ---- FIX 2: 合并 system prompt + memory + 当前 user 到同一个 User message ----
        full_user_content = ""

        # 1) system prompt
        if self.system_prompt:
            full_user_content += f"[System Prompt]: {self.system_prompt}\n\n"

        # 2) memory
        mem = self.memory.get_all()
        if mem:
            full_user_content += "[Memory]:\n"
            for m in mem:
                full_user_content += f"- {m}\n"
            full_user_content += "\n"

        # 3) image placeholder
        if image_paths:
            full_user_content += "<image_placeholder> "

        # 4) 当前用户问题
        full_user_content += user_query

        # ---- FIX 3: 只有一条 User + 一条 Assistant ----
        conversation = [
            {
                "role":  "<|User|>",
                "content": full_user_content,
                "images": image_paths if image_paths else []
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        return conversation


    def run(self, user_query: str, image_path=None, args=None):

        # deepseek 要求 images 是 list
        image_paths = [image_path] if image_path else None

        conversation = self.build_conversation(user_query, image_paths)

        # load images
        pil_images = load_pil_images(conversation)

        prepare_inputs = self.vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
        ).to(self.vl_gpt.device, dtype=self.dtype)

        with torch.no_grad():

            if args.chunk_size == -1:
                inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                # incremental_prefilling when using 40G GPU for vl2-small
                inputs_embeds, past_key_values = self.vl_gpt.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=args.chunk_size
                )

            # run the model to get the response
            outputs = self.vl_gpt.generate(
                # inputs_embeds=inputs_embeds[:, -1:],
                # input_ids=prepare_inputs.input_ids[:, -1:],
                inputs_embeds=inputs_embeds,
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                past_key_values=past_key_values,

                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,

                # do_sample=False,
                # repetition_penalty=1.1,

                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,

                use_cache=True,
            )

            answer = self.tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
            print(f"{prepare_inputs['sft_format'][0]}", answer)

            vg_image = parse_ref_bbox(answer, image=pil_images[-1])
            if vg_image is not None:
                vg_image.save("./vg.jpg", format="JPEG", quality=85)

        # update memory
        self.memory.add(user_query)

        return answer


if __name__ == '__main__':

    from llm.prompts import trading_prompt

    parser = ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()
    
    model_path = "./models/deepseek-vl2-tiny"
    llm = initial_llm(model_path)


    answer = llm.run("you are a stock manager.", 
                     image_path="./picture/show/us/BA/kline_0.png",
                     args=args)
                     
    print("=== Done ===")
    print(answer)