"""
Test Database - Check if detections are being saved
"""

import psycopg2
from datetime import datetime

# Database connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="waste_system",
    user="postgres",
    password="baohuy2501"
)

cur = conn.cursor()

print("=" * 80)
print("🔍 CHECKING DATABASE RECORDS")
print("=" * 80)

# Check detection sessions
print("\n📊 DETECTION SESSIONS:")
cur.execute("""
    SELECT id, started_at, ended_at, total_detections, 
           organic_count, recyclable_count, hazardous_count, other_count
    FROM detection_sessions
    ORDER BY started_at DESC
    LIMIT 10
""")
sessions = cur.fetchall()
if sessions:
    print(f"\nFound {len(sessions)} session(s):")
    for s in sessions:
        print(f"  Session #{s[0]}: Started={s[1]}, Ended={s[2]}")
        print(f"    Total: {s[3]}, Organic: {s[4]}, Recyclable: {s[5]}, Hazardous: {s[6]}, Other: {s[7]}")
else:
    print("  No sessions found")

# Check detections
print("\n🎯 DETECTIONS:")
cur.execute("""
    SELECT COUNT(*) FROM detections
""")
total = cur.fetchone()[0]
print(f"\nTotal detections: {total}")

if total > 0:
    cur.execute("""
        SELECT d.id, d.session_id, d.label, d.category, d.confidence, 
               d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height, d.detected_at, d.tracking_data
        FROM detections d
        ORDER BY d.detected_at DESC
        LIMIT 20
    """)
    detections = cur.fetchall()
    print(f"\nLast {len(detections)} detection(s):")
    for d in detections:
        print(f"  ID #{d[0]}: {d[2]} ({d[3]}) - Confidence: {d[4]:.2f}")
        print(f"    Session: {d[1]}, BBox: [{d[5]:.1f}, {d[6]:.1f}, {d[7]:.1f}, {d[8]:.1f}]")
        print(f"    Time: {d[9]}")
        if d[10]:  # tracking_data
            import json
            meta = json.loads(d[10]) if isinstance(d[10], str) else d[10]
            print(f"    📊 Tracking: Duration={meta.get('duration_seconds', 'N/A')}s, "
                  f"Frames={meta.get('frame_count', 'N/A')}, "
                  f"AvgConf={meta.get('average_confidence', 'N/A')}")
        print()

# Check by category
print("\n📈 DETECTIONS BY CATEGORY:")
cur.execute("""
    SELECT category, COUNT(*) as count
    FROM detections
    GROUP BY category
    ORDER BY count DESC
""")
categories = cur.fetchall()
if categories:
    for cat in categories:
        print(f"  {cat[0]}: {cat[1]} detections")
else:
    print("  No detections yet")

# Check waste bins
print("\n🗑️ WASTE BINS:")
cur.execute("""
    SELECT COUNT(*) FROM waste_bins
""")
bins_total = cur.fetchone()[0]
print(f"Total bins: {bins_total}")

if bins_total > 0:
    cur.execute("""
        SELECT id, name, category, latitude, longitude, is_active
        FROM waste_bins
        LIMIT 10
    """)
    bins = cur.fetchall()
    for b in bins:
        status = "Active" if b[5] else "Inactive"
        print(f"  Bin #{b[0]}: {b[1]} ({b[2]}) - {status}")
        print(f"    Location: ({b[3]}, {b[4]})")

print("\n" + "=" * 80)
print("✅ Database check complete!")
print("=" * 80)

cur.close()
conn.close()
