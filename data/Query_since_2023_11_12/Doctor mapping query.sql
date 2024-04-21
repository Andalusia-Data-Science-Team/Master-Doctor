--billing
select distinct o.OrganizationUnifiedName,s.Staff_name from dax.factbilling b inner join dax.dimstaff s 
on b.Doctor_Key=s.Staff_Key
inner join dax.dimOrganization o on b.Organization_Key=o.OrganizationKey
where
service_date>cast(DATEADD(month, DATEDIFF(month, -1, getdate()) - 2, 0)as date)
and  b.organization_key in (1,3,5,4,9,10,13)
order by o.OrganizationUnifiedName

--CRM
select distinct Source_Unit, Topic from dimlead
where Source_Unit in ('HJH','ALW','ADC','APC','AKW')
and Lead_Source in ('DoctorUna','EasyDoc','Tebcan','Vezeeta')
and Lead_CreationDate>cast(DATEADD(mm, DATEDIFF(m,0,GETDATE()),0)as date)
order by Source_Unit
