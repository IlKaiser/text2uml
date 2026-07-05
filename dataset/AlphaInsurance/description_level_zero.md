## Classes
- Alpha Insurance — insurance company, provides insurance policies, help desk, head office
- Customer — the client, addresses it, profile (assessed, deemed trustworthy), account
- Broker — first account manager, registered in the system, assesses profile on the spot or sends file to head office
- Insurance Policy — type(s) of insurance product, price (monthly or yearly invoicing)
- Contract/Offer — preliminary offer on an insurance product (in person or by email), signed by both parties
- Invoice — monthly or yearly, based on the product price
- Claim — claim for compensation, sent on the insured event
- Claim Case — one or several (e.g. material damage & physical damage separately), case file, approved or not
- Estimator — different, by area of expertise
- Report — issued by them, stored in the database (at least one year after payment, legal purposes)
- Compensation Decision — registered on approval, stipulates eligible costs for (partial) refund, sum of compensation
- Document — supplied, basis of compensation calculation

## Relationships
- Alpha Insurance and Insurance Policy, it provides various types of them (1 -> 0..*).
- Customer and Broker, it assigned to follow its file, tracing it as its first account manager (1 -> 1).
- Customer and Insurance Policy, it indicates type(s) of them it would like to sign for (1 -> 1..*).
- Broker and Contract/Offer, it makes a preliminary one (in person or by email, also to existing ones) after it assesses its profile and deems it trustworthy (1 -> 0..*).
- Customer and Contract/Offer, it signs it (both parties) on its agreement to it (1 -> 0..*).
- Contract/Offer and Invoice, its coverage invoiced through them (monthly or yearly, per the choice made in it) (1 -> 0..*).
- Customer and Claim, it should send one of them for compensation on the insured event (1 -> 0..*).
- Claim and Claim Case, it opens one or several of them (e.g. accident: material & physical damage separately) (1 -> 1..*).
- Claim Case and Estimator, its complete case file sent to them for assessment (by their area of expertise) (1 -> 1..*).
- Estimator and Report, they issue them, its approval decided per them (1 -> 0..*).
- Claim Case and Compensation Decision, it registered on its approval (stipulating eligible costs for its partial refund) (1 -> 0..1).
- Compensation Decision and Document, it draws on the supplied ones, its sum calculated from them and paid to their account (1 -> 1..*).
