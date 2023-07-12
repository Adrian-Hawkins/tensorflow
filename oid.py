from bson import ObjectId

# string_id = "6163c5d90b81234c7dd0e0e5"  # Example string representation of ObjectId
string_id = '64a6fc6af9bd4ecf8fa78343'
object_id = ObjectId(string_id)
print(object_id)

# Now `object_id` is an actual ObjectId that you can use in MongoDB queries
