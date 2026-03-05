# How to Book Parking

## Reservation Channel

All reservations are handled through the parking chatbot. The chatbot manages the full reservation flow and collects required details from the user in sequence.

## Information the Chatbot Requests

The chatbot asks for:
- Customer name
- Parking facility
- Date (YYYY-MM-DD)
- Start time (HH:MM, 24-hour format)
- Duration in hours

## Validation Checks Performed by the Chatbot

The chatbot validates each required field:
- Customer name must be non-empty
- Facility must match an available parking facility
- Date must be today or later
- Start time must be a valid 24-hour time
- Duration must be a whole number from 1 to 168 hours

If any field is invalid or missing, the chatbot requests that field again before moving forward.

## When Reservation Is Created

The reservation is created only after all required fields are present and valid. Newly created reservations are marked as pending admin review.

## What Counts as a Successful Booking

A booking is considered successful only after an administrator reviews the reservation and confirms it.

## Confirmation Format

After admin confirmation, the chatbot returns a final confirmation message with:
- Reservation ID
- Facility
- Date and start time
- Duration
- Final status: confirmed
