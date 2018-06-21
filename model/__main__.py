from cortex_client import InputMessage, OutputMessage


# the entry point of your model
def main(params):
    msg = InputMessage.from_params(params)

    input = msg.payload.get('text')

    return OutputMessage.create().with_payload(
        {
            'text':  input
        }
    ).to_params()

