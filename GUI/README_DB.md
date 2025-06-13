┌────────────┐
│   users    │
├────────────┤
│ id (PK)    │
│ username   │
│ password   │
│ full_name  │
│ role       │
│ created_at │
└────┬───────┘
     │
     ▼
┌────────────┐            ┌────────────────────┐
│  cameras   │───────────▶│   vehicle_stats     │◀────────────┐
├────────────┤            ├────────────────────┤             │
│ id (PK)    │            │ id (PK)            │             │
│ name       │            │ camera_id (FK)     │             │
│ location   │            │ date (DATE)        │             │
│ installed  │            │ time_slot (0–47)   │             │
└────┬───────┘            │ direction          │             │
     │                    │ vehicle_type       │             │
     │                    │ vehicle_count      │             │
     │                    └────────────────────┘             │
     │                                                       │
     ▼                                                       │
┌────────────────────┐                                       │
│    avg_speeds      │                                       │
├────────────────────┤                                       │
│ id (PK)            │                                       │
│ camera_id (FK)     │                                       │
│ date (DATE)        │                                       │
│ time_slot (0–47)   │                                       │
│ average_speed      │                                       │
└────────────────────┘                                       │
     ▲                                                       │
     │                                                       │
     │                                                       ▼
┌────────────────────┐                            ┌────────────────────────────┐
│   camera_area      │                            │ daily_traffic_summary      │
├────────────────────┤                            ├────────────────────────────┤
│ id (PK)            │                            │ id (PK)                    │
│ camera_id (FK)     │                            │ camera_id (FK)             │
│ area_id (FK)       │                            │ date (DATE)                │
│ location_detail    │                            │ total_vehicle_count        │
└────────┬───────────┘                            │ avg_speed                  │
         │                                        │ peak_time_slot             │
         ▼                                        │ direction_with_most_traffic│
┌────────────────────┐                            └────────────────────────────┘
│      areas         │
├────────────────────┤
│ id (PK)            │
│ name               │
│ description        │
└────────────────────┘

             (Optional)
                   ▲
                   │
                   ▼
        ┌────────────────────┐
        │   traffic_events   │
        ├────────────────────┤
        │ id (PK)            │
        │ camera_id (FK)     │
        │ event_time         │
        │ event_type         │
        │ description        │
        └────────────────────┘
