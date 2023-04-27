import os
from typing import List

from cog import BasePredictor, Input, Path
from huggingface_hub import login


from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

# stage 1

prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'

# text embeds


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        login(token=os.environ['HUGGINGFACE_KEY'])
        self._generator = torch.manual_seed(0)
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_1.enable_model_cpu_offload()
        self._stage_1 = stage_1

        stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_2.enable_model_cpu_offload()
        self._stage_2 = stage_2

        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
        stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_3.enable_model_cpu_offload()

        self._stage_3 = stage_3

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says 'very deep learning'",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),


    ) -> List[Path]:
        """Run a single prediction on the model"""
        prompt_embeds, negative_embeds = self._stage_1.encode_prompt(prompt)

        image = self._stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=self._generator, output_type="pt").images
        pt_to_pil(image)[0].save("./if_stage_I.png")

        image = self._stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=self._generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_II.png")

        image = self._stage_3(prompt=prompt, image=image, generator=self._generator, noise_level=100).images
        image[0].save("./if_stage_III.png")

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        image = self._stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=self._generator, output_type="pt").images
        stage_1_path = "/tmp/if_stage_I.png"
        pt_to_pil(image)[0].save(stage_1_path)

        image = self._stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=self._generator, output_type="pt"
        ).images
        stage_2_path = "/tmp/if_stage_II.png"
        pt_to_pil(image)[0].save(stage_2_path)

        image = self._stage_3(prompt=prompt, image=image, generator=self._generator, noise_level=100).images
        stage_3_path = "/tmp/if_stage_III.png"
        image[0].save(stage_3_path)
        return [Path(stage_1_path), Path(stage_2_path), Path(stage_3_path)]

