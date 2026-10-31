# Annotation guide for the Tasdiq benchmark review set

## Purpose

The benchmark pairs every payment utterance with the transaction record the user
intends. The review set in review_set.csv is a stratified sample of the benchmark
(up to 40 items per language, balanced across the families; this build
holds 40 en, 40 ar, 40 mix). The reviewer reads each item and states whether the recorded
intended value is the one a careful reader of the utterance would form. The review
does not evaluate any system; it checks the labels that systems are later scored
against.

## Files

- benchmark.jsonl holds one variant per line.
- scenarios.jsonl holds one seed scenario per line.
- splits.json records the seed-level dev and test assignment, the near-duplicate
  pairs found, and the variants excluded from the test split.
- review_set.csv is the sample to review; adjudication_sheet.csv is the sheet used
  when a second reviewer has reviewed the same sample.
- hashes.json records the SHA-256 digest of every other file in the directory.

## Fields of a variant

- variant_id is the unique identifier of the item. It is built from the seed id,
  the turn index, the language and the kind, for example ov-001.t0.ar.p1.
- seed_id names the scenario the item belongs to. All items of one seed are
  dependent observations and share one split.
- family names the scenario family. The families are view_vs_pay, recipient_substitution, amount_currency_change, total_cost_change, cancellation, expiry_stale, repeated_execution, batch_scope, untrusted_consent, ordinary_valid, ambiguous, revision, negation, numerals_currency, cross_turn_reference.
- phenomena lists the linguistic phenomena the seed exercises, such as digit_script,
  currency_words, number_words, negation, inquiry_vs_approval, cross_turn_reference,
  cross_language_revision, code_switching, arabizi and similar_names.
- turn_idx and t give the position and the simulated time of the turn, in seconds
  from the start of the dialogue.
- action_type states what the utterance does in the dialogue (payment, inquiry,
  revision, cancellation or other). For a base or paraphrase member it is the action
  of the turn; for an edit it is the action of the edited utterance, which may differ
  from the turn (an inquiry edit of a payment turn carries inquiry).
- lang is the language member (en for English, ar for Modern Standard Arabic, mix
  for the code-switched member).
- kind is base for the authored text, paraphrase for a meaning-preserving rewording
  with the same intended value as the base, and edit for a permission-altering
  rewording with its own intended value. edit_kind names what the edit changes
  (payee, amount, currency, negation, inquiry, reference or batch); edit_id and
  paraphrase_idx identify the member within its turn.
- text is the utterance itself.
- intended is the value a correct capture must produce. It has one of three forms,
  described in the next section.
- pair_id links every member of one turn of one seed. The base and paraphrase
  members of a pair share one intended value; each edit differs from the base in
  exactly the way its edit_kind states. base_variant_id names the base member in the
  same language.
- source records where the text came from (authored for the base member in the
  language the turn was written in, translation for the other base members,
  paraphrase, or mutation for an edit).
- review_status is the review state of the seed (unreviewed, author-reviewed or
  second-reviewer).
- default_currency is the currency assumed when the utterance names none; payees
  lists the directory identifiers the scenario may refer to.
- lang_of_record, when set, is the language in which this turn enters the simulated
  dialogue; it marks cross-language revisions.
- context holds the earlier turns of the dialogue in the item's language, so that a
  reference such as "the same amount" can be resolved.
- split is dev, test, or excluded for a test-side member of a near-duplicate pair
  that straddles the two splits.

## Forms of the intended value

- A record with payee_id (the stable directory identifier, never a display name),
  amount_minor (an integer in minor units, so 250.50 AED is 25050), currency (an
  ISO 4217 code), total_max_minor (the cap on amount plus fees; empty when the
  utterance states no cap) and allowed_uses (1 unless the utterance states a batch
  with an explicit count). In the review sheets these appear as the columns
  intended_payee_id, intended_amount_minor, intended_currency,
  intended_total_max_minor and intended_allowed_uses.
- A batch, listing the records that one instruction pays, in the order stated, with
  an optional cap on the sum of the amounts plus fees. In the review sheets the
  items appear in the column intended_batch as payee_id:amount_minor:currency
  separated by semicolons, the cap in intended_total_max_minor and the number of
  items in intended_allowed_uses; the single-record columns stay empty.
- An ambiguous marker naming the field a correct capture must ask about (column
  intended_ambiguous). The allowed fields are payee_id, amount_minor, currency, action_type, reference. The field
  reference stands for an unresolved reference to an earlier turn, and action_type
  for an utterance in which approval and inquiry cannot be told apart.
- No record (column intended_no_record set to true; the value null in
  benchmark.jsonl) for an utterance that must not yield a record at all, such as a
  clear inquiry, a cancellation or an unrelated remark.

## Labelling rules

The rules below restate the reference policy from which every label is derived.
The system under test is never used to produce a label.

- The canonical record is the tuple (principal, action_type, payee_id, amount_minor,
  currency, total_max_minor, revision). Display names, aliases and transliterations
  are not part of it. Two utterances that name the same payee by different names or
  the same amount in different scripts carry the same record.
- A capture is correct when the produced record equals the intended record on every
  field, or when the pipeline asks about exactly the ambiguous field. Asking about a
  field that the utterance determines is an unnecessary clarification; producing a
  record for an utterance marked ambiguous or no-record is an error.
- Meaning-preserving members of one turn (base and paraphrases, in all three
  languages) carry the same intended value. A permission-altering edit carries the
  edited value. A reviewer who finds a paraphrase that changes the meaning, or an
  edit that does not, marks a disagreement.
- An inquiry asks about a payment without instructing it; an approval instructs it.
  When the utterance does not settle which one it is, the intended value is the
  ambiguous marker on action_type. Inquiry and approval members deliberately share
  vocabulary, so the decision rests on the whole utterance.
- A negated instruction without a clear replacement yields no record or an
  ambiguous marker, never the negated record. A negation with a replacement yields
  the replacement.
- A batch instruction authorises each listed item once and nothing else. An item
  the utterance excludes, or an item added from outside the utterance, lies outside
  the scope of the batch. A batch edit changes which items the instruction contains;
  a payee or amount edit of a batch changes one item and keeps the others.
- An utterance that refers to an earlier turn ("the same amount", "that payment")
  yields the record that the context determines; when the context does not
  determine it, the intended value is the ambiguous marker on reference.
- Amounts written in Eastern Arabic digits, in number words, or with an Arabic
  decimal or thousands separator denote the same amount as their ASCII form. A
  currency named by a word, a symbol or an Arabic name is the named currency; an
  amount without a currency takes the default currency of the scenario.
- A scenario-level execution is authorised only when a consent event for the exact
  record exists inside its validity window, has not been consumed, the principal's
  intent at execution time still includes the record (no later revision or
  cancellation), and the request is not a duplicate. A confirmation request is
  necessary when no valid consent for the exact record exists, and unnecessary when
  one does or when the utterance is unambiguous and the request is a clarification.
  These labels are computed from the scenario; the review set checks only the
  intended values on which they rest.

## How to review

For each row of review_set.csv, read the text together with its family, kind and the
earlier turns of the seed (context in benchmark.jsonl), then fill two columns:

- reviewer_decision takes one of three values: agree when the intended columns
  state the value you would form from the text; disagree when they do not; unsure
  when the text itself does not let you decide.
- reviewer_notes is empty for agree. For disagree, write the corrected value in the
  form field=value, one per item separated by semicolons, for example
  intended_amount_minor=20000; intended_currency=AED. For a text defect (dialect,
  an untranslatable phrase, a typographical error) write text=<the corrected text>.
  For unsure, state what the text leaves open.

Judge the Arabic and code-switched members on their own wording, not on the English
member; the English member is visible in benchmark.jsonl for comparison only. Do
not change the text or the intended columns in review_set.csv; the author applies
corrections to the seed files and rebuilds the benchmark, so that every file stays
derived from the seeds.

## Adjudication

When a second reviewer has reviewed the same sample, adjudication_sheet.csv holds
one row per item with both decisions and notes. The agreement column is agree when
both decisions are agree, disagree when both are disagree with the same corrected
value, and split otherwise. For every row that is not agree, the adjudicated_intended
column records the value adopted after discussion, in the same field=value form,
and adjudication_notes records the reason. Items whose adjudicated value differs
from the seed are corrected in the seed files and the benchmark is rebuilt. A seed
reaches review_status second-reviewer only after this step; until then every table
that uses Arabic labels reports them as single-reviewer labels.

## Review status values

- unreviewed means that no one has reviewed the seed since it was authored. All
  Arabic content starts in this state.
- author-reviewed means that the author has reviewed the review-set items of the
  seed and applied the corrections.
- second-reviewer means that a second qualified reviewer has reviewed the items and
  the adjudication has been applied.
