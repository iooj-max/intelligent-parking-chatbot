# How to Book Airport Parking

## Reservation Channel

All parking reservations are created through the parking chatbot. The chatbot guides the user step by step and keeps the reservation request in progress until all required details are collected and validated.

## Information the Chatbot Requests

To start and complete a reservation request, the chatbot asks for:
- Customer name
- Parking facility
- Date (YYYY-MM-DD)
- Start time (HH:MM, 24-hour format)
- Duration in hours

## Validation Checks Performed by the Chatbot

Before creating a reservation, the chatbot validates:
- Customer name is not empty
- Parking facility exists in available facilities
- Date is today or a future date
- Start time is a valid 24-hour time
- Duration is an integer between 1 and 168 hours

If a value is missing or invalid, the chatbot asks for one field at a time until the request is complete and valid.

## When Reservation Is Created

A reservation record is created only after all required fields are collected and all checks pass. At creation time, the reservation status is set to pending admin review.

## What Counts as a Successful Booking

A booking is successful only when an administrator verifies the request and approves it. A created reservation that is still pending review is not yet a successful booking.

## Confirmation Format

After admin approval, the user receives a confirmation message that includes:
- Reservation ID
- Facility
- Date and start time
- Duration
- Final status: confirmed
