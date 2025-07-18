from slime.utils.types import Sample

async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    data = []
    pendings = []
    while len(data) < target_data_size:
        pendings += _submit_generate_tasks(min_size=target_data_size - len(pendings))

        # wait for the generation to finish
        raw_done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        filtered_done = _postprocess_done_data(raw_done, max_num_outputs=target_data_size-len(data))

        pbar.update(args.n_samples_per_prompt * len(filtered_done))
        data += filtered_done

    await abort(pendings)
    ...

def _submit_generate_tasks(min_size: int):
    pendings = []
    # this part is copied from genereate_rollout_async
    while len(pendings) < min_size:
        # get samples from the buffer and submit the generation requests.
        samples = data_buffer.get_samples(args.over_sampling_batch_size)

        # this part is copied from State.submit_generate_tasks
        for group in samples:
            pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )

def _postprocess_done_data(done, max_num_outputs):
    data = []

    for task in done:
        group: list[Sample] = task.result()

        if do_print:
            print(
                f"First rollout sample: {[group[0].prompt + group[0].response]}, label: {group[0].label}, reward: {group[0].reward}",
                flush=True,
            )
            do_print = False

        assert len(group) == args.n_samples_per_prompt
        if dynamic_filter is not None and not dynamic_filter(args, group):
            # state.remaining_batch_size -= 1 # <-- no need for this
            continue

        # add the samples to the data
        # NOTE: here we have not stored all the unused samples back to the data buffer.
        if len(data) >= max_num_outputs:
            continue

        data.append(group)

    return data