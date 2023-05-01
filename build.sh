cog build
docker build -t r8.im/andremoeller/deep-floyd --build-arg HUGGINGFACE_API_KEY=$HUGGINGFACE_API_TOKEN .
docker push r8.im/andremoeller/deep-floyd
