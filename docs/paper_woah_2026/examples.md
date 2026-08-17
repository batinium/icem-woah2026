# Synthetic Release Examples

These examples are synthetic schematics for explaining I-CEM. They are not raw
dataset rows and do not reproduce real social-media posts, slurs, or profanity.
They follow WOAH's reporting guidance by using placeholders instead of explicit
offensive content.

| Length | Synthetic source | Importance only | Fixed window | I-CEM |
| --- | --- | --- | --- | --- |
| Short | `[PERSON] wrote: please do not call [GROUP] [SLUR].` | `[SLUR]` | `[GROUP] [SLUR]` | `do not call [GROUP] [SLUR]` |
| Medium | `At the forum, [PERSON] quoted "[GROUP] are [SLUR]" and replied that the claim is hateful.` | `[GROUP] ... [SLUR]` | `quoted "[GROUP] are [SLUR]"` | `quoted "[GROUP] are [SLUR]" and replied that the claim is hateful` |
| Long | `After the meeting, [PERSON] wrote that the speaker did not say [GROUP] are [SLUR]; the post criticized that rumor as harmful.` | `[GROUP] ... [SLUR]` | `say [GROUP] are [SLUR]` | `did not say [GROUP] are [SLUR]; the post criticized that rumor as harmful` |

What these examples show:

- `top_k` evidence can retain the strongest harm cue while stripping away the
  stance needed to interpret it.
- Fixed windows recover nearby tokens but can still miss negation, quotation, or
  counterspeech cues.
- I-CEM expands around retained evidence when context rules detect target,
  harm, negation, quotation/reporting, counterspeech, or stance relationships.
