import xml.etree.ElementTree as ET

defaults = {"fs.defaultFS": "file:///", "dfs.replication": "3", "dfs.blocksize": "134217728",
            "mapreduce.framework.name": "local", "yarn.nodemanager.resource.memory-mb": "8192"}

site_xml = {
 "core-site.xml": """<configuration><property><name>fs.defaultFS</name>
     <value>hdfs://localhost:9000</value></property></configuration>""",
 "hdfs-site.xml": """<configuration><property><name>dfs.replication</name><value>1</value></property>
     </configuration>""",
 "mapred-site.xml": """<configuration><property><name>mapreduce.framework.name</name>
     <value>yarn</value></property></configuration>""",
}

effective = dict(defaults)
source = {k: "default" for k in defaults}
for fname, text in site_xml.items():
    for prop in ET.fromstring(text).findall("property"):
        name, value = prop.findtext("name").strip(), prop.findtext("value").strip()
        effective[name], source[name] = value, fname

for k in effective:
    print(f"{k:<38} = {effective[k]:<24} ({source[k]})")
print("\nblock size in MB:", int(effective["dfs.blocksize"]) // (1024 * 1024))
